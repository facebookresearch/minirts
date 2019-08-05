# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from common_utils import assert_eq


class ContSoftmaxSampler:
    def __init__(self, cont_key, cont_prob_key, key, prob_key):
        self.cont_key = cont_key
        self.cont_prob_key = cont_prob_key
        self.key = key
        self.prob_key = prob_key
        self.ref_net = None

    def add_ref_net(self, ref_net):
        self.ref_net = ref_net

    def clamp_prob(self, probs, min_prob):
        clamped_probs = {
            self.cont_prob_key: probs[self.cont_prob_key].clamp(min_prob, 1 - min_prob),
            self.prob_key: probs[self.prob_key].clamp(min_prob, 1 - min_prob)
        }
        return clamped_probs

    def sample(self, cont_probs, probs, prev_samples):
        """Categorical sampler that can persist the previous samples
        with some probability.

        Args:
            cont_probs: probabilities of keeping the previous samples
                    [batch, 2]
            probs: probabilities returned by model forward
                    [batch, num_actions]
            prev_samples: previosly sampled actions
                    [batch]
        return:
            cont_samples: [batch]
            samples: [batch]
        """
        assert_eq(cont_probs.size(1), 2)
        cont_samples = cont_probs.multinomial(1).squeeze(1)
        new_samples = probs.multinomial(1).squeeze(1)

        assert_eq(prev_samples.size(), new_samples.size())
        samples = cont_samples * prev_samples + (1 - cont_samples) * new_samples
        return {
            self.cont_key: cont_samples,
            self.key: samples,
        }

    def get_log_prob(self, probs, samples):
        """Compute log prob of given samples

        Args:
            probs: {
                self.cont_prob_key: [batch, 2],
                self.prob_key [batch, num_actions]
            }
            samples: {
                self.cont_key: [batch],
                self.key: [batch]
            }
        return: p: [batch]
        """
        cont_probs = probs[self.cont_prob_key]
        probs_ = probs[self.prob_key]
        cont_samples = samples[self.cont_key]#.float()
        samples_ = samples[self.key].clamp(min=0, max=probs_.size(0) - 1)

        # assert_eq(cont_samples.size(1), 1)
        # assert_eq(samples_.size(1), 1)

        prob = probs_.gather(1, samples_).squeeze(1)
        # cont_prob = cont_probs.gather(1, cont_samples.unsqueeze(1)).squeeze(1)
        cont_samples = cont_samples.squeeze(1).float()
        final_prob = (cont_samples * cont_probs[:, 1]
                      + (1 - cont_samples) * prob * cont_probs[:, 0])
        # print('>>>', final_prob.size())
        return final_prob.log()

    def get_entropy(self, probs):
        """Compute entropy for ContSoftmaxSampler

        Args:
            probs: {
                self.cont_prob_key: [batch, 2],
                self.prob_key [batch, num_actions]
            }
        """
        cont_probs = probs[self.cont_prob_key]
        probs_ = probs[self.prob_key]
        # cont_entropy = -(cont_probs * cont_probs.log()).sum(1)
        entropy = -(probs_ * probs_.log()).sum(1)
        # total_entropy = cont_entropy + cont_probs[:, 0] * entropy
        # return total_entropy
        return entropy

    def get_true_entropy(self, probs):
        """Compute entropy for ContSoftmaxSampler

        Args:
            probs: {
                self.cont_prob_key: [batch, 2],
                self.prob_key [batch, num_actions]
            }
        """
        cont_probs = probs[self.cont_prob_key]
        probs_ = probs[self.prob_key]
        cont_entropy = -(cont_probs * cont_probs.log()).sum(1)
        entropy = -(probs_ * probs_.log()).sum(1)
        total_entropy = cont_entropy + cont_probs[:, 0] * entropy
        return total_entropy

    def get_cross_entropy(self, probs, batch, min_prob):
        ref_probs = self.ref_net(batch)#['pi']
        ref_probs = self.clamp_prob(ref_probs, min_prob)

        contp = probs[self.cont_prob_key]
        softp = probs[self.prob_key] * contp[:, 0].unsqueeze(1)

        ref_contp = ref_probs[self.cont_prob_key].detach()
        ref_softp = ref_probs[self.prob_key].detach() * ref_contp[:, 0].unsqueeze(1)

        cont_xent = ref_contp[:, 1] * contp[:, 1].log()
        soft_xent = (ref_softp * softp.log()).sum(1)
        # print(cont_xent.size(), soft_xent.size())
        xent = -(cont_xent + soft_xent) #).sum(1)

        ref_ent = self.get_true_entropy(ref_probs)
        kl = xent - ref_ent

        # if kl.max() > 1e-4:
        #     import pdb; pdb.set_trace()

        return xent, kl


if __name__ == '__main__':
    import torch

    cont_probs = torch.rand(10, 1)
    cont_probs = torch.cat([1 - cont_probs, cont_probs], 1)
    probs = torch.nn.functional.softmax(torch.rand(10, 30), dim=1)

    prev_samples = torch.randint(0, 30, (10,)).long()

    sampler = ContSoftmaxSampler('cont', 'cont_pi', 'sample', 'pi')
    ts_samples = sampler.sample(cont_probs, probs, prev_samples)

    ts_probs = {
        'cont_pi': cont_probs,
        'pi': probs
    }

    logp = sampler.get_log_prob(ts_probs, ts_samples)
    ent = sampler.get_entropy(ts_probs)

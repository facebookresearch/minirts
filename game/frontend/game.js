// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//

var scale = function(x) {
  var m = x / 2;
  return m + Math.floor((x - m) * SCALER);
}

var is_coach_intf = function() {
  return window.player_type == "coach";
}

var is_player_intf = function() {
  return window.player_type == "player";
}

var is_spectator_intf = function() {
  return !is_coach_intf() && !is_player_intf();
}

var is_player = function(player) {
  return player.player_id === 0;
}

var is_coach = function(player) {
  return player.player_id === 1;
}

var set_cursor = function(cursor) {
  window.canvas.style.cursor = cursor;
  document.body.style.cursor = cursor;
}

var get_cursor = function() {
  return window.canvas.style.cursor;
}

var notify = function(text) {
  var audio = document.getElementById("audio");
  audio.play();
}

var make_spectator_intf = function() {
  window.fps_range = document.createElement("INPUT");
  window.fps_range.type = "range";
  window.fps_range.min = window.min_speed;
  window.fps_range.max = window.max_speed;
  window.fps_range.value = 0;
  window.fps_range.step = 1;
  window.fps_range.style.position = "absolute";
  window.fps_range.style.top = 320;
  window.fps_range.style.left = window.left_frame_width + 50;
  window.fps_range.style.zindex = 2;
  window.fps_range.style.width = "300px";
  window.fps_range.style.height = "30px";
  window.fps_range.oninput = function(){
    var update = this.value - window.speed;
    window.speed = this.value;
    if (update > 0) {
      for (var i = 0; i < update; i++){
        send_cmd(window.tick + " F");
      }
    }
    if (update < 0) {
      for (var i = 0; i < -update; i++){
        send_cmd(window.tick + " W");
      }
    }
  }

  document.body.appendChild(window.fps_range);
  var button_left = window.left_frame_width + scale(50);

  document.getElementById("cmd_hint").style.display = "none";

  var addButton = function(text, cmd) {
    var button = document.createElement("button");
    button.innerHTML = text;
    button.style.position = "absolute";
    button.style.top = 200;
    button.style.left = button_left;
    button.style.zindex = 2;
    button.style.width = "60px";
    button.style.height = "30px";
    button_left += 100;
    document.body.appendChild(button);
    button.addEventListener ("click", function() {
      if (cmd == "F") {
        if (window.speed >= window.max_speed) return;
        else {
          window.speed = window.speed + 1;
          window.fps_range.value = window.speed;
        }
      }
      if (cmd == "W") {
        if (window.speed <= window.min_speed) return;
        else {
          window.speed = window.speed - 1;
          window.fps_range.value = window.speed;
        }
      }
      send_cmd(window.tick + " " + cmd);
    });
    return button;
  };

  window.button_faster = addButton("Faster", "F");
  window.button_slower = addButton("Slower", "W");
  window.button_cycle = addButton("Cycle", "C");
  window.button_pause = addButton("Pause", "P");

  window.progress_range = document.createElement("INPUT");
  window.progress_range.type = "range";
  window.progress_range.min = 0;
  window.progress_range.max = 100;
  window.progress_range.value = 0;
  window.progress_range.step = 1;
  window.progress_range.style.position = "absolute";
  window.progress_range.style.top = 420;
  window.progress_range.style.left = window.left_frame_width + 50;
  window.progress_range.style.zindex = 2;
  window.progress_range.style.width = "300px";
  window.progress_range.style.height = "30px";
  window.progress_range.oninput = function(){
    send_cmd(window.tick + " S " + this.value);
  };
  document.body.appendChild(window.progress_range);

  window.cmd_input = document.getElementById("cmd_input");
  window.cmd_input.value = "";
  window.cmd_input.readOnly = "true";
  window.cmd_input.style.resize = "none";
  window.cmd_input.disabled = "true";

  document.getElementById("cmd_input_label").innerHTML = "Current order to execute on:";
  window.cmd_input.style.color = "red";

  window.cmd_history = document.getElementById("cmd_history");
  window.cmd_history.value = "";
  window.cmd_history.readOnly = "true";
  window.cmd_history.style.resize = "none";
  window.cmd_history.disabled = "true";

  window.cmd_button = document.getElementById("finish_btn");
};


var send_cmd = function(s) {
  window.dealer.send(s);
};

var parse_inst = function(inst) {
  var pieces = inst.split(" ");
  var any_minus = false;
  var res = "";
  for (var i in pieces) {
    var p = pieces[i];
    if (p === "-1") {
      any_minus = true;
      break;
    }
    res += String.fromCharCode(parseInt(p, 10));
  }
  if (!any_minus) return inst;
  return res;
}

var draw_instructions = function(instructions) {
  if (instructions === null) {
    return;
  }
  var history = "";
  for (var i in instructions) {
    var inst = instructions[i];
    if (inst["done"] === true) {
      var item = i + ": " + parse_inst(inst["text"]);
      history = item + "\n" + history;
    }
  }
  window.cmd_history.value = history;
  var cur_inst = "";
  var warn = false;
  for (var i in instructions) {
    var inst = instructions[i];
    if (i > instructions.length - 3 && inst["warn"]) {
      warn = true;
    }
    if (inst["done"] === false) {
      cur_inst = parse_inst(inst["text"]);
    }
  }
  if (is_player_intf()) {
    if (window.num_instructions < instructions.length) {
      notify("You have a new order!");
      swal("You have a new order, follow it precisely:", cur_inst, "warning")
        .then((value) => {
          send_cmd(window.tick + " A " + window.cmd_input.value);
        });
    }
    if (warn) {
      document.getElementById("cmd_warn_label").style.display = "inline";
    } else {
      document.getElementById("cmd_warn_label").style.display = "none";
    }
  }
  else if (is_spectator_intf()) {
    if (warn) {
      document.getElementById("cmd_warn_label").style.display = "inline";
    } else {
      document.getElementById("cmd_warn_label").style.display = "none";
    }
  }
  window.cmd_input.value = cur_inst;
};

var resize = function() {
  var min_w = 300;
  var min_h = 300;
  var max_w = 700;
  var max_h = 700;
  var min_cz = 10;
  var max_cz = 20;
  var dw = 200;
  var dh = 200;
  var w = Math.min(Math.max(window.innerWidth - dw, min_w), max_w);
  var h = Math.min(Math.max(window.innerHeight - dh, min_h), max_h);
  var pw = 1.0 * (w - min_w) / (max_w - min_w);
  var ph = 1.0 * (h - min_h) / (max_h - min_h);
  window.SCALER = Math.min(pw, ph);
  window.cell_size = min_cz + Math.floor(window.SCALER * (max_cz - min_cz));

  var extra = scale(300);
  if (is_coach_intf()) {
    extra = scale(200);
  }
  window.canvas.width = window.map_x * window.cell_size + extra;
  window.canvas.height = window.map_y * window.cell_size;
  window.left_frame_width = window.map_x * window.cell_size;

  if (window.cmd_input != null) {
    window.cmd_input.style.width = String(window.map_x * window.cell_size) + "px";
    window.cmd_history.style.width = String(window.map_x * window.cell_size) + "px";
  }

  if (is_spectator_intf()) {
    if (window.button_faster != null) {
      window.button_faster.style.left = window.left_frame_width + scale(50) + scale(0 * 100);
      window.button_slower.style.left = window.left_frame_width + scale(50) + scale(1 * 100);
      window.button_cycle.style.left = window.left_frame_width + scale(50) + scale(2 * 100);
      window.button_pause.style.left = window.left_frame_width + scale(50) + scale(3 * 100);
      window.progress_range.style.left = window.left_frame_width + scale(50);
      window.fps_range.style.left = window.left_frame_width + scale(50);
    }
  }
}

var make_cursor = function(color) {
  var cursor = document.createElement('canvas');
  var ctx = cursor.getContext('2d');

  cursor.width = 16;
  cursor.height = 16;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(8, 8, 7, 0, 2 * Math.PI, false);
  ctx.moveTo(0, 8);
  ctx.lineTo(15, 8);
  ctx.moveTo(8, 0);
  ctx.lineTo(8, 15);
  ctx.stroke();
  ctx.closePath();
  return 'url(' + cursor.toDataURL() + '), auto';
}

var make_build_cursor = function(image) {
  var cursor = document.createElement('canvas');
  var ctx = cursor.getContext('2d');

  var w = 40;
  var h = 40;
  cursor.width = w;
  cursor.height = h;
  ctx.globalAlpha = 0.5;
  ctx.drawImage(image, 0, 0, w, h);

  return 'url(' + cursor.toDataURL() + ') 20 20, auto';
}

var send_instruction = function() {
  if (window.cmd_input.value.length > 1) {
    send_cmd(window.tick + ' X ' + window.cmd_input.value);
    window.cmd_history.value = window.inst_id + ": " + window.cmd_input.value + "\n" + window.cmd_history.value;
    window.inst_id = window.inst_id + 1;
    window.curr_inst = window.cmd_input.value;
    window.cmd_input.value = "";
  }
}

var make_coach_intf = function() {
  window.cmd_input = document.getElementById("cmd_input");
  window.cmd_input.value = "";
  window.cmd_input.style.resize = "none";
  window.cmd_input.addEventListener ("keydown", function(e) {
    if (e.keyCode === 13) {
      e.preventDefault();
      send_instruction();
    }
  });

  document.getElementById("cmd_input_label").innerHTML = "Enter your order here:";

  window.cmd_history = document.getElementById("cmd_history");
  window.cmd_history.value = "";
  window.cmd_history.readOnly = "true";
  window.cmd_history.style.resize = "none";
  window.cmd_history.disabled = "true";

  window.cmd_button = document.getElementById("issue_btn");
  window.cmd_button.addEventListener("click", function() {
    if (window.cmd_input.value != "") {
      send_instruction();
    }
  });
  document.getElementById("issue_btn_div").style.display = "inline";

  window.cmd_inter = document.getElementById("stop_btn");
  window.cmd_inter.addEventListener("click", function() {
    send_cmd(window.tick + ' I ' + window.curr_inst);
    window.was_stopped = true;
  });
  document.getElementById("stop_btn_div").style.display = "inline";

  window.cmd_warn = document.getElementById("warn_btn");
  window.cmd_warn.addEventListener("click", function() {
    send_cmd(window.tick + ' Q ');
  });
  document.getElementById("warn_btn_div").style.display = "inline";

};


var make_player_intf = function() {
  window.cmd_input = document.getElementById("cmd_input");
  window.cmd_input.value = "";
  window.cmd_input.readOnly = "true";
  window.cmd_input.style.resize = "none";
  window.cmd_input.disabled = "true";

  document.getElementById("cmd_hint").style.display = "none";

  document.getElementById("cmd_input_label").innerHTML = "Current order to execute on:";
  window.cmd_input.style.color = "red";

  window.cmd_history = document.getElementById("cmd_history");
  window.cmd_history.value = "";
  window.cmd_history.readOnly = "true";
  window.cmd_history.style.resize = "none";
  window.cmd_history.disabled = "true";

  window.cmd_button = document.getElementById("finish_btn");
  window.cmd_button.addEventListener("click", function() {
    send_cmd(window.tick + ' Z ' + window.cmd_input.value);
  });
  document.getElementById("finish_btn_div").style.display = "inline";
};


var are_unit_types_selected = function(types) {
  if (window.last_state === null) return false;
  if (is_spectator_intf()) return false;
  var selected = window.last_state.selected_units;
  if (!selected) return false;
  var any = false;
  for (var i in window.last_state.units) {
    var unit = window.last_state.units[i];
    if (selected.indexOf(unit.id) >= 0) {
      if (types.indexOf(unit.unit_type) >= 0) {
        any = true;
      } else {
        return false;
      }
    }
  }
  return any;
}

var are_workers_selected = function() {
  var worker_ty = [window.unit_id["PEASANT"]];
  return are_unit_types_selected(worker_ty);
}

var are_units_selected = function() {
  var unit_ty = [
    window.unit_id["PEASANT"],
    window.unit_id["SWORDMAN"],
    window.unit_id["SPEARMAN"],
    window.unit_id["CAVALRY"],
    window.unit_id["ARCHER"],
    window.unit_id["DRAGON"],
    window.unit_id["CATAPULT"]];
  return are_unit_types_selected(unit_ty);
}

var are_towers_selected = function() {
  var tower_ty = [window.unit_id["GUARD_TOWER"]];
  return are_unit_types_selected(tower_ty);
}

var get_unit_type = function(id) {
  if (window.last_state === null) return -1;
  for (var i in window.last_state.units) {
    if (id === window.last_state.units[i].id) {
      return window.last_state.units[i].unit_type;
    }
  }
  return -1;
}

var is_build_cmd_allowed = function(key, types) {
  if (window.last_state === null) return false;
  var selected = window.last_state.selected_units;
  if (!selected) return false;
  if (selected.length != 1) return false;
  var id = selected[0];
  var ty = get_unit_type(id);
  if (types.indexOf(ty) < 0) {
    return false;
  }
  var def = window.last_state.gamedef.units[ty];
  for (var i in def.build_skills) {
    if (key == def.build_skills[i].hotkey) {
      return true;
    }
  }
  return false;
}

var is_worker_cmd_allowed = function(key) {
  var worker_types = [window.unit_id["PEASANT"]];
  return is_build_cmd_allowed(key, worker_types);
}

var is_building_cmd_allowed = function(key) {
  var building_types = [
    window.unit_id["BARRACK"],
    window.unit_id["BLACKSMITH"],
    window.unit_id["STABLE"],
    window.unit_id["WORKSHOP"],
    window.unit_id["TOWN_HALL"]];
  return is_build_cmd_allowed(key, building_types);
}

var on_map = function(m) {
  var counter = 0;
  for (var y = 0; y < m.height; y++) {
    for (var x = 0; x < m.width; x++) {
      var type = m.slots[counter];
      var seen_before = false;
      if (m.seen_terrain != null) {
        seen_before = m.seen_terrain[counter];
      }
      var spec = window.terrain_sprites[window.terrains[type]];
      var x1 = x * window.cell_size + window.cell_size / 2;
      var y1 = y * window.cell_size + window.cell_size / 2;
      draw_terrain_sprite(spec, x1, y1, seen_before);
      counter += 1;
    }
  }
}

var draw_hp = function(bbox, states, font_color, player_color, fill_color, progress, draw_str) {
  var x1 = bbox[0];
  var y1 = bbox[1];
  var x2 = bbox[2];
  var y2 = bbox[3];
  var hp_ratio = states[0];
  var state_str = states[1];
  var margin = scale(3);
  window.ctx.fillStyle = "black";
  window.ctx.lineWidth = margin;
  window.ctx.beginPath();
  window.ctx.rect(x1, y1, x2 - x1, y2 - y1);
  window.ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  window.ctx.strokeStyle = player_color;
  window.ctx.stroke();
  window.ctx.closePath();
  var color = fill_color;
  if (progress && hp_ratio <= 0.5) color = "yellow";
  if (progress && hp_ratio <= 0.2) color = "red";
  window.ctx.fillStyle = color;
  window.ctx.fillRect(x1, y1, Math.floor((x2 - x1) * hp_ratio + 0.5), y2 - y1);
  if (draw_str && state_str) {
    window.ctx.beginPath();
    window.ctx.fillStyle = font_color;
    window.ctx.font = "10px Arial";
    window.ctx.fillText(state_str, x2 + 10, y1 + cell_size * 0.5);
    window.ctx.closePath();
  }
}

var on_unit = function(u, isSelected, isAttacking) {
  var player_color = window.player_colors[u.player_id];
  var sprites = window.player_sprites[player_color];
  var p =  u.p;
  var last_p = u.last_p;
  var xy = convert_xy(p.x, p.y);

  var unit_name = window.unit_names_minirts[u.unit_type];
  var spec = sprites[unit_name];
  if (isAttacking && (unit_name != "GUARD_TOWER")) {
    draw_sprites_attack(spec, xy[0], xy[1], null);
  } else {
    draw_sprites(spec, xy[0], xy[1], null, u.temporary);
  }

  var hp_ratio = u.hp / u.max_hp;
  var state_str;
  if ("cmd" in u) {
    if (u.cmd.cmd[0] != 'I') {
        state_str = u.cmd.cmd[0] + u.cmd.state;
    }
  }
  var sw = Math.floor(window.cell_size * spec["_select_scale"]);
  var sh = Math.floor(window.cell_size * spec["_select_scale"]);
  var x1 = xy[0] - Math.floor(sw * 0.4);
  var y1 = xy[1] - sh / 2 - 10;
  var x2 = x1 + Math.floor(sw * 0.8);
  var y2 = y1 + scale(5);
  if (u.unit_type == window.unit_id["RESOURCE"]) {
    var unit_color = "black";
  } else {
    var unit_color = player_color;
  }

  draw_hp([x1, y1, x2, y2], [hp_ratio, state_str], "white", unit_color, "green", true, false);
  if (isSelected) {
    window.ctx.lineWidth = 2;
    window.ctx.beginPath();
    window.ctx.rect(xy[0] - sw / 2 - scale(2), xy[1] - sh / 2 - scale(2), sw + scale(4), sh + scale(4));
    window.ctx.strokeStyle = player_color;
    window.ctx.stroke();
    window.ctx.closePath();
  }
};

var on_bullet = function(bullet) {
  var xy = convert_xy(bullet.p.x, bullet.p.y);
  draw_bullet(bullets, xy[0], xy[1], bullet.state);
}

var pad_zero = function(x) {
  if (x < 10) {
    return "0" + String(x);
  }
  return String(x);
}

var on_players_stats = function(players, game) {
  var x1 = left_frame_width;
  var y1 = 0;
  var label = "";
  if (players.length === 1) {
    label = "MINERALS: " + players[0].resource;
  } else {
    for (var i in players) {
      var player = players[i];
      label = label + "PLAYER" + player.player_id + ":" + player.resource + "  ";
    }
  }
  window.ctx.beginPath()
  window.ctx.fillStyle = "Black";
  window.ctx.font = "13px Arial";
  if (is_spectator_intf()) {
    window.ctx.fillText("TICK: " + game.tick, x1 + window.cell_size, y1 + window.cell_size / 2 + 5);
  } else {
    var diff_ms = ((new Date()) - start_time) / 1000;
    var sec = Math.floor(diff_ms % 60);
    var min = Math.floor((diff_ms / 60) % 60);
    window.ctx.fillText("TIME: " + pad_zero(min) + ":" + pad_zero(sec), x1 + window.cell_size, y1 + window.cell_size / 2 + 5);
  }

  y1 += scale(30);
  window.ctx.fillStyle = "green";
  window.ctx.font = "bold 13px Arial";
  window.ctx.fillText(label, x1 + window.cell_size, y1 + window.cell_size / 2 + scale(5));
  window.ctx.closePath();
}

// Draw units that have been seen.
var on_player_seen_units = function(m) {
  if ("units" in m) {
    var oldAlpha = window.ctx.globalAlpha;
    window.ctx.globalAlpha = 0.5;

    for (var i in m.units) {
      on_unit(m.units[i], false);
    }
    window.ctx.globalAlpha = oldAlpha;
  }
}

var draw_state = function(u, game) {
  var resource = game.players[0].resource;
  var player_color = window.player_colors[u.player_id];
  var sprites = window.player_sprites[window.player_colors[u.player_id]];
  var x1 = window.left_frame_width + scale(20);
  var y1 = scale(60);
  var spec = sprites[window.unit_names_minirts[u.unit_type]];
  draw_sprites(spec, x1 + window.cell_size / 2, y1 + window.cell_size, null, false);

  window.ctx.beginPath();
  if (is_spectator_intf()) {
    var title = window.unit_names_minirts[u.unit_type] + ': ' + u.cmd.cmd + '[' + u.cmd.state + ']';
  } else {
    var title = window.unit_names_minirts[u.unit_type];
  }
  window.ctx.fillStyle = "black";
  window.ctx.font = "10px Arial";
  window.ctx.fillText(title, x1 + 1.5 * cell_size + scale(5), y1 + cell_size / 2 + scale(5), scale(300));
  window.ctx.closePath();
  var ratio = u.hp / u.max_hp;
  var label = "HP: " + u.hp + " / " + u.max_hp;
  var x2 = x1 + 1.5 * window.cell_size + scale(5);
  var y2 = y1 + window.cell_size / 2 + scale(20);
  if (u.unit_type == window.unit_id["RESOURCE"]) {
    var unit_color = "black";
  } else {
    var unit_color = player_color;
  }
  draw_hp([x2, y2, x2 + scale(100), y2 + scale(15)], [ratio, label], 'black', unit_color, 'green', true, true);
  if (u.player_id != game.player_id) {
    return;
  }

  var x3 = x1;
  var y3 = y1 + 2 * window.cell_size + 10;
  for (var i in u.cds) {
    var cd = u.cds[i];
    if (cd.cd > 0) {
      var curr = Math.min(game.tick - cd.last, cd.cd);
      if (cd.last === 0) curr = cd.cd;
      ratio = curr / cd.cd;
      var label = cd.name.split("_")[1] + " COOLDOWN"
      draw_hp([x3, y3, x3 + scale(100), y3 + scale(15)], [ratio, label], 'black', player_color, 'magenta', false, true);
      y3 += scale(20);
    }
  }

  // building is not ready yet
  if (u.temporary) {
    return;
  }

  var cmd_names = {
    200: ["ATTACK", "a"],
    201: ["MOVE", "m"],
    202: ["BUILD","b"],
    203: ["GATHER", "g"]
  };
  var def = game.gamedef.units[u.unit_type];
  var title = "[HOTKEY]COMMAND:";
  for (var i in def.allowed_cmds) {
    var cmd = def.allowed_cmds[i];
    var name = cmd_names[cmd.id][0];
    if (name == "BUILD") continue;
    var hotkey = cmd_names[cmd.id][1];
    title = title + ' [' + hotkey + "]" + name;
  }
  window.ctx.beginPath()
  window.ctx.fillStyle = "black";
  window.ctx.font = "bold 15px Arial";
  y3 += scale(40);
  ctx.fillText("Use buttons below to build units:", x3, y3);
  y3 += scale(15);
  window.ctx.fillStyle = "red";
  window.ctx.font = "13px Arial";
  ctx.fillText("1) LEFT CLICK on a button", x3, y3);
  y3 += scale(15);
  ctx.fillText("2) Then RIGHT CLICK on the map", x3, y3);
  window.ctx.closePath();
  y3 += scale(15);

  for (var i in def.build_skills) {
    var skill = def.build_skills[i];
    var unit_name = window.unit_names_minirts[skill.unit_type];
    var spec = sprites[unit_name];

    var disabled = skill.price > resource;
    draw_button_sprites(skill.hotkey, disabled, spec, x3 + 40 / 2, y3 + 40 / 2, 1);
    window.ctx.beginPath();
    var title = unit_name + " [" + skill.price + " MINERALS]";
    window.ctx.fillStyle = "black";
    window.ctx.font = "12px Arial";
    window.ctx.fillText(title, x3 + 40 + 10, y3 + 40 / 2 + 5);
    window.ctx.closePath();
    y3 += 40 + 10;
  }
}

var convert_xy = function(x, y){
  var xc = Math.floor(x * cell_size + cell_size / 2);
  var yc = Math.floor(y * cell_size + cell_size / 2);
  return [xc, yc]
}

var convert_xy_back = function(x, y){
  var xx = x / cell_size - 0.5;
  var yy = y / cell_size - 0.5;
  return [xx, yy];
}

var load_sprites = function(spec) {
  // Default behavior.
  var specReady = false;
  var specImage = new Image();
  specImage.onload = function () {
    specReady = true;
  };
  specImage.src = spec["_file"];
  specImage.crossOrigin = "Anonymous";
  spec["image"] = specImage;

  if (spec["_file_attack"] != null) {
    var specAttackImage = new Image();
    specAttackImage.onload = function () {
        specReady = true;
    };
    specAttackImage.src = spec["_file_attack"];
    spec["image_attack"] = specAttackImage;
  } else {
    spec["image_attack"] = null;
  }
  return spec;
}

var load_player_sprites = function(player) {
  var sprites = {};
  sprites["RESOURCE"] = load_sprites({
    "_file" : "imgs/mineral1.png",
    "_file_attack": null,
    "_scale": 1.2,
    "_select_scale" : 1
  });
  sprites["PEASANT"] = load_sprites({
    "_file": "rts/medieval/" + player + "/peasant.png",
    "_file_attack": "rts/medieval/" + player + "/peasant_attack.png",
    "_scale": 1.5,
    "_select_scale" : 1
  });
  sprites["SWORDMAN"] = load_sprites({
    "_file": "rts/medieval/" + player + "/swordman.png",
    "_file_attack": "rts/medieval/" + player + "/swordman_attack.png",
    "_scale": 1.5,
    "_select_scale" : 1
  });
  sprites["SPEARMAN"] = load_sprites({
    "_file": "rts/medieval/" + player + "/spearman.png",
    "_file_attack": "rts/medieval/" + player + "/spearman_attack.png",
    "_scale": 1.5,
    "_select_scale" : 1
  });
  sprites["CAVALRY"] = load_sprites({
    "_file": "rts/medieval/" + player + "/cavalry.png",
    "_file_attack": "rts/medieval/" + player + "/cavalry_attack.png",
    "_scale": 1.5,
    "_select_scale" : 1
  });
  sprites["ARCHER"] = load_sprites({
    "_file": "rts/medieval/" + player + "/archer.png",
    "_file_attack": "rts/medieval/" + player + "/archer_attack.png",
    "_scale": 1.5,
    "_select_scale" : 1
  });
  sprites["DRAGON"] = load_sprites({
    "_file": "rts/medieval/" + player + "/dragon.png",
    "_file_attack": "rts/medieval/" + player + "/dragon_attack.png",
    "_scale": 2,
    "_select_scale" : 1.5
  });
  sprites["CATAPULT"] = load_sprites({
    "_file": "rts/medieval/" + player + "/catapult.png",
    "_file_attack": "rts/medieval/" + player + "/catapult_attack.png",
    "_scale": 2,
    "_select_scale" : 1.5
  });
  sprites["BARRACK"] = load_sprites({
    "_file": "rts/medieval/" + player + "/barrack.png",
    "_file_attack": null,
    "_scale": 2,
    "_select_scale" : 1.2
  });
  sprites["BLACKSMITH"] = load_sprites({
    "_file": "rts/medieval/" + player + "/blacksmith.png",
    "_file_attack": null,
    "_scale": 2,
    "_select_scale" : 1.2
  });
  sprites["STABLE"] = load_sprites({
    "_file": "rts/medieval/" + player + "/stable.png",
    "_file_attack": null,
    "_scale": 2,
    "_select_scale" : 1.2
  });
  sprites["WORKSHOP"] = load_sprites({
    "_file": "rts/medieval/" + player + "/workshop.png",
    "_file_attack": null,
    "_scale": 2,
    "_select_scale" : 1.2
  });
  sprites["AVIARY"] = load_sprites({
    "_file": "rts/medieval/" + player + "/aviary.png",
    "_file_attack": null,
    "_scale": 2,
    "_select_scale" : 1.2
  });
  sprites["ARCHERY"] = load_sprites({
    "_file": "rts/medieval/" + player + "/archery.png",
    "_file_attack": null,
    "_scale": 2,
    "_select_scale" : 1.2
  });
  sprites["GUARD_TOWER"] = load_sprites({
    "_file": "rts/medieval/" + player + "/guard_tower.png",
    "_file_attack": null,
    "_scale": 1.5,
    "_select_scale" : 1
  });
  sprites["TOWN_HALL"] = load_sprites({
    "_file": "rts/medieval/" + player + "/town_hall.png",
    "_file_attack": null,
    "_scale": 2,
    "_select_scale" : 1.2
  });
  return sprites;
}

var draw_bullet = function(spec, px, py, ori) {
  var image = spec["image"]
  var width = image.width;
  var height = image.height;
  if (!("_sizes" in spec)) {
    window.ctx.drawImage(image, px - width / 2, py - height / 2);
  } else {
    var sw = spec["_sizes"][0];
    var sh = spec["_sizes"][1];
    var xidx = spec[ori][0];
    var yidx = spec[ori][1];
    var cx = xidx[Math.floor(window.tick / 3) % xidx.length] * sw;
    var cy = yidx[Math.floor(window.tick / 3) % yidx.length] * sh;
    window.ctx.drawImage(image, cx + 2, cy + 2, sw - 4, sh - 4, px - sw / 2, py - sh / 2, sw - 4, sh - 4);
  }
}

var draw_sprites = function(spec, px, py, scale, temporary) {
  var image = spec["image"];
  if (scale === null) {
    scale = spec["_scale"];
  }
  var w = Math.floor(cell_size * scale);
  var h = Math.floor(cell_size * scale);
  if (temporary) {
    var old = window.ctx.globalAlpha;
    window.ctx.globalAlpha = 0.5;
    window.ctx.drawImage(image, px - w / 2, py - 0.7 * h, w, h);
    window.ctx.globalAlpha = old;
  } else {
    window.ctx.drawImage(image, px - w / 2, py - 0.7 * h, w, h);
  }
}

var draw_sprites_attack = function(spec, px, py, scale) {
  var image = spec["image_attack"];
  if (scale === null) {
    scale = spec["_scale"];
  }
  var w = Math.floor(cell_size * scale);
  var h = Math.floor(cell_size * scale);
  window.ctx.drawImage(image, px - w / 2, py - 0.7 * h, w, h);
}

var get_mouse_pos = function(e) {
  return {x: e.pageX - window.canvas.offsetLeft, y: e.pageY - window.canvas.offsetTop};
}

var is_inside = function(p, rect) {
  return p.x > rect.x && p.x < rect.x + rect.width && p.y < rect.y + rect.height && p.y > rect.y;
}

var is_over_build_buttons = function(e) {
  var p = get_mouse_pos(e);
  for (var i = 0; i < window.build_buttons.length; i++) {
    var data = window.build_buttons[i];
    if (is_inside(p, data.rect)) {
      return true;
    }
  }
  return false;
}

var build_callback = function(e) {
  var p = get_mouse_pos(e);
  for (var i = 0; i < window.build_buttons.length; i++) {
    var data = window.build_buttons[i];
    if (data.disabled) {
      continue;
    }
    if (is_inside(p, data.rect)) {
      if (is_worker_cmd_allowed(data.hotkey)) {
        set_cursor(data.cursor);
        window.building = true;
        send_cmd(window.tick + " " + data.hotkey);
      } else if (is_building_cmd_allowed(data.hotkey)) {
        send_cmd(window.tick + " " + data.hotkey);
      }
      return true;
    }
  }
  return false;
}

var draw_mouse = function(mouse) {
  var draw_box = function(x1, y1, x2, y2, color) {
    window.ctx.lineWidth = 2;
    window.ctx.beginPath();
    window.ctx.rect(x1, y1, x2 - x1, y2 - y1);
    window.ctx.strokeStyle = color;
    window.ctx.stroke();
    window.ctx.closePath();
  };

  var draw_click = function(x, y, ty) {
    window.ctx.strokeStyle = "white";
    window.ctx.fillStyle = "white";
    window.ctx.lineWidth = 2;
    window.ctx.beginPath();
    window.ctx.arc(x, y, 7, 0, 2 * Math.PI, false);
    window.ctx.moveTo(x - 8, y);
    window.ctx.lineTo(x + 7, y);
    window.ctx.moveTo(x, y - 8);
    window.ctx.lineTo(x, y + 7);
    window.ctx.font = "bold 16px Arial";
    window.ctx.fillText(ty, x + 10, y, 20);
    window.ctx.stroke();
    window.ctx.closePath();
  }

  if (is_spectator_intf()) {
    if (mouse) {
      if (mouse.act == "B") {
        var xy1 = convert_xy(mouse.x1, mouse.y1);
        var xy2 = convert_xy(mouse.x2, mouse.y2);
        draw_box(xy1[0], xy1[1], xy2[0], xy2[1], "white");
      } else {
        var xy1 = convert_xy(mouse.x1, mouse.y1);
        draw_click(xy1[0], xy1[1], mouse.act);
      }
    }
  } else {
    if (window.dragging && window.x_down && window.y_down) {
      draw_box(window.x_down, window.y_down, window.x_curr, window.y_curr, "green");
    }
  }
}

var draw_button_sprites = function(hotkey, disabled, spec, px, py, scale) {
  var image = spec["image"];
  if (scale === null) {
    scale = spec["_scale"];
  }
  var w = 40;
  var h = 40;
  var margin = 2;
  var rect = {x: px - w / 2 - margin, y: py - h / 2 - margin, width: w + 2 * margin, height: h + 2 * margin};
  var cursor = make_build_cursor(image);
  window.build_buttons.push({rect: rect, hotkey: hotkey, disabled: disabled, cursor: cursor});
  var old = window.ctx.globalAlpha;
  if (disabled) {
    window.ctx.globalAlpha = 0.5;
  }

  window.ctx.drawImage(image, px - w / 2, py - h / 2, w, h);
  window.ctx.strokeStyle = "black";
  window.ctx.lineWidth = 2;
  window.ctx.beginPath();
  window.ctx.moveTo(px - w / 2, py - h / 2);
  window.ctx.lineTo(px + w / 2, py - h / 2);
  window.ctx.lineTo(px + w / 2, py + h / 2);
  window.ctx.lineTo(px - w / 2, py + h / 2);
  window.ctx.lineTo(px - w / 2, py - h / 2);
  window.ctx.stroke();
  window.ctx.closePath();

  if (disabled) {
    window.ctx.globalAlpha = old;
  }
}

var draw_terrain_sprite = function(spec, px, py, seen_before) {
  var fog_image = window.terrain_sprites["FOG"]["image"];
  var image = spec["image"];
  if (seen_before) {
    window.ctx.drawImage(fog_image, px - window.cell_size / 2, py - window.cell_size / 2, window.cell_size, window.cell_size);
    var oldAlpha = window.ctx.globalAlpha;
    window.ctx.globalAlpha = 0.3;
    window.ctx.drawImage(image, px - window.cell_size / 2, py - window.cell_size / 2, window.cell_size, window.cell_size);
    window.ctx.globalAlpha = oldAlpha;
  } else {
    window.ctx.drawImage(image, px - window.cell_size / 2, py - window.cell_size / 2, window.cell_size, window.cell_size);
  }
}

var render = function (game) {
  window.map_x = game.rts_map.width;
  window.map_y = game.rts_map.height;
  window.tick = game.tick;
  window.game_status = game.game_status;
  resize();

  if (game.winner != -1) {
    if (game.player_id == game.winner) {
      swal("Congratulations!", "You won the game!", "success");
    } else {
      swal("You lost!", "Thank you for playing our game, best of luck next time!", "error");
    }
  }

  if (window.render_memo) {
    window.render_memo = false;
    fill_units_table(game);
    fill_buildings_table(game);
    show_interface();
  }

  update_units_table(game, game.players[0].resource);
  update_buildings_table(game, game.players[0].resource);

  if (window.building === false) {
    if (are_units_selected()) {
      if (get_cursor() === "default") {
        set_cursor(window.move_cursor);
      }
    } else if (are_towers_selected()) {
      if (get_cursor() === "default") {
        set_cursor(window.attack_cursor);
      }
    } else if (document.body.style.cursor !== "pointer") {
      set_cursor("default");
    }
  }

  on_map(game.rts_map);
  if (!game.spectator) {
    on_player_seen_units(game.rts_map);
  }

  var all_units = {};
  var selected = {};
  on_players_stats(game.players, game);

  var attacking = {};
  for (var i in game.bullets) {
    var b = game.bullets[i];
    if (b.state.indexOf("CREATE") >= 0) {
      attacking[b.id_from] = b;
    }
  }
  var disabled_bullets = {};

  for (var i in game.units) {
    var unit = game.units[i];
    all_units[unit.id] = unit;

    var s_units = game.selected_units;
    var isSelected = (s_units && s_units.indexOf(unit.id) >= 0);
    if (isSelected) {
      selected[unit.id] = unit;
    }
    var isAttacking = (unit.id in attacking);

    // use attacking animation for gathering
    if (unit.cmd.cmd === "GATHER" && unit.cmd.state == 1) {
      isAttacking = true;
    }

    on_unit(unit, isSelected, isAttacking);

    if (window.melee_units.indexOf(window.unit_names_minirts[unit.unit_type]) >= 0) {
      disabled_bullets[unit.id] = true;
    }
  }
  draw_mouse(game.mouse);
  for (var i in game.bullets) {
    var b = game.bullets[i];
    if (!(b.id_from in disabled_bullets)) {
      on_bullet(b);
    }
  }

  if (is_player_intf() || is_spectator_intf()) {
    draw_instructions(game["instructions"]);
    if (game["instructions"] == null) {
      window.num_instructions = 0;
    } else {
      window.num_instructions = game["instructions"].length;
    }
  }

  window.build_buttons = [];

  if (!is_spectator_intf()) {
    var len = Object.keys(selected).length;
    if (len == 1) {
      var idx = Object.keys(selected)[0];
      var unit = selected[idx];
      draw_state(unit, game);
    }
  }
  window.ctx.beginPath();
  window.ctx.fillStyle = "black";
  window.ctx.font = "10px Arial";
  if (len > 1) {
    var label = len + " units";
    window.ctx.fillText(label, window.left_frame_width + scale(50), scale(200));
  }
  if (is_spectator_intf()) {
    var label = "Current FPS is " + Math.floor((scale(50)) * Math.pow(1.3, speed));
    window.ctx.fillText(label, window.left_frame_width + scale(50), 300);

    if (game.replay_length) {
      progress_range.value = 100 * game.tick / game.replay_length;
    }

    var label = "Current progress_percent is " + window.progress_range.value;
    window.ctx.fillText(label, window.left_frame_width + scale(50), 400);
  }

  if (game.game_status != 0) {
    window.ctx.fillStyle = "white";
    window.ctx.font = "30px Arial";
    if (is_coach_intf()) {
      if (game.game_status == 2) {
        var label =  "Game is paused: please issue an order.";
        var label2 = "Your partner cannot take any actions now.";
      } else {
        var label = "Game is paused: wait for order acceptance.";
        var label2 = "After this you can issue another command.";
      }
    } else {
      var label = "Game is paused: please wait for an order.";
      var label2 = "You cannot take any actions at this point.";
    }
    window.ctx.fillText(label, Math.floor(0.5 * window.map_x * window.cell_size) - scale(275), Math.floor(0.5 * window.map_y * window.cell_size));
    window.ctx.fillText(label2, Math.floor(0.5 * window.map_x * window.cell_size) - scale(275), 30 + Math.floor(0.5 * window.map_y * window.cell_size));
    if (window.frozen_init) {
      window.frozen_init = false;
      if (is_coach_intf()) {
        if (window.tick < 5) {
          swal("Please issue an order for your partner.", "", "warning");
        } else {
          if (game.game_status == 2 && !window.was_stopped) {
            swal("The previous order is finished.\n Please issue a new order.", "", "warning");
            notify("Please issue a new order.");
            window.was_stopped = false;
          }
        }
      }
    }
  } else {
    window.frozen_init = true;
  }

  if (is_coach_intf()) {
    window.cmd_inter.disabled = game.game_status != 0;
    window.cmd_warn.disabled = game.game_status != 0;
  } else {
    window.cmd_button.disabled = game.game_status != 0;
  }

  window.ctx.closePath();
};

var init = function() {
  // request notification
  if (Notification) {
    if (Notification.permission !== "granted") {
      Notification.requestPermission();
    }
  }
  // Create the canvas
  window.canvas = document.getElementById("canvas");
  window.ctx = window.canvas.getContext("2d");
  // max sizes
  window.map_x = 40;
  window.map_y = 30;
  window.SCALER = 1.0;
  window.cell_size = 35;
  window.inst_id = 0;
  window.cmd_input = null;
  window.cmd_hsitory = null;
  window.cmd_button = null;
  window.cmd_inter = null;
  window.cmd_warn = null;
  window.cmd_warn_label = null;
  window.progress_range = null;
  window.fps_range = null;
  window.start_time = null;
  window.curr_inst = "";
  window.button_faster = null;
  window.button_slower = null;
  window.button_cycle = null;
  window.button_pause = null;
  window.build_buttons = [];
  window.host = "minirts.net";
  window.player_type = "player";
  window.frozen_init = true;
  window.render_memo = true;
  window.num_instructions = 0;
  window.game_status = 0;
  window.was_stopped = false;

  window.canvas.width = window.map_x * window.cell_size + 600;
  window.canvas.height = window.map_y * window.cell_size + 200;
  window.left_frame_width = window.map_x * window.cell_size;
  window.player_colors = ['blue', 'red', 'red'];

  window.terrains = ["SOIL", "SAND", "GRASS", "ROCK", "WATER", "FOG"];
  window.unit_names_minirts = [
    "RESOURCE",
    "PEASANT",
    "SPEARMAN",
    "SWORDMAN",
    "CAVALRY",
    "DRAGON",
    "ARCHER",
    "CATAPULT",
    "BARRACK",
    "BLACKSMITH",
    "STABLE",
    "WORKSHOP",
    "AVIARY",
    "ARCHERY",
    "GUARD_TOWER",
    "TOWN_HALL"
  ];
  window.unit_id = {};
  window.unit_names_minirts.forEach(function (value, id) {
    window.unit_id[value] = id;
  });

  window.x_down = null;
  window.y_down = null;
  window.x_curr = 0;
  window.y_curr = 0;
  window.dragging = false;
  window.building = false;
  window.tick = 0;
  window.dealer = null;
  window.speed = 0;
  window.min_speed = -10;
  window.max_speed = 5;
  window.last_state = null;
  window.melee_units = [
    "PEASANT",
    "SWORDMAN",
    "CAVALRY"
  ];

  window.attack_cursor = make_cursor('red');
  window.move_cursor = make_cursor('green');
  window.gather_cursor = make_cursor('cyan');

  window.canvas.oncontextmenu = function (e) {
    e.preventDefault();
  };

  window.canvas.addEventListener("mousedown", function (e) {
    if (e.button === 0) {
      if (build_callback(e)) return;
      var xy0 = convert_xy_back(e.pageX, e.pageY);
      if (xy0[0] > window.map_x || xy0[1] > window.map_y) return;
      window.x_down = e.pageX;
      window.y_down = e.pageY;
    }
  }, false);

  window.canvas.addEventListener("mouseup", function (e) {
    if (is_over_build_buttons(e)) return;
    var xy0 = convert_xy_back(e.pageX, e.pageY);

    if (e.button === 0) {
      var xy = convert_xy_back(x_down, y_down);
      if (dragging && x_down && y_down) {
        if (e.pageX < window.canvas.width && e.pageY < window.canvas.height) {
          xy0[0] = Math.min(xy0[0], window.map_x - 1);
          xy0[1] = Math.min(xy0[1], window.map_y - 1);
          send_cmd([window.tick, "B", xy[0], xy[1], xy0[0], xy0[1]].join(" "));
        }
      } else {
        send_cmd([window.tick, "L", xy[0], xy[1]].join(" "));
      }
      x_down = null;
      y_down = null;
      dragging = false;
      set_cursor("default");
      building = false;
    }
    if (e.button === 2) {
      send_cmd([window.tick, "R", xy0[0], xy0[1]].join(" "));
      if (building) {
        set_cursor("default");
      }
      building = false;
    }
  }, false);


  window.addEventListener("mouseup", function (e) {
    var xy0 = convert_xy_back(e.pageX, e.pageY);
    if (e.button == 0) {
      if (e.pageX >= window.canvas.width || e.pageY >= window.canvas.height) {
        if (dragging && x_down && y_down) {
          var xy = convert_xy_back(x_down, y_down);
          xy0[0] = Math.min(xy0[0], window.map_x - 1);
          xy0[1] = Math.min(xy0[1], window.map_y - 1);
          send_cmd([window.tick, "B", xy[0], xy[1], xy0[0], xy0[1]].join(" "));
        }
        x_down = null;
        y_down = null;
        dragging = false;

        set_cursor("default");
        building = false;
      }
    }
  }, false);



  window.canvas.addEventListener("mousemove", function (e) {
    if (is_over_build_buttons(e) && !window.building) {
      set_cursor("pointer");
      return;
    }
    // change cursor
    var cursor = move_cursor;
    if (last_state != null) {
      var xy = convert_xy_back(e.pageX, e.pageY);
      var x = xy[0];
      var y = xy[1];
      for (var i in last_state.units) {
        var unit = last_state.units[i];
        var dist = Math.sqrt((x - unit.p.x) * (x - unit.p.x) + (y - unit.p.y) * (y - unit.p.y));
        if (dist < 0.35) {
          if (unit.unit_type === unit_id["RESOURCE"]) {
            cursor = window.gather_cursor;
          }
          if (unit.unit_type != unit_id["RESOURCE"] && unit.player_id != last_state.player_id && !unit.temporary) {
            cursor = window.attack_cursor;
          }
        }
      }
    }

    var xy = convert_xy_back(e.pageX, e.pageY);
    if (building === false) {
      if (are_units_selected()) {
        set_cursor(cursor);
      } else if (are_towers_selected()) {
        set_cursor(window.attack_cursor);
      } else {
        set_cursor("default");
      }
    }

    if (x_down && y_down) {
      x_curr = Math.max(0, Math.min(e.pageX, window.map_x * window.cell_size));
      y_curr = Math.max(0, Math.min(e.pageY, window.map_y * window.cell_size));
      var diffx = x_down - x_curr;
      var diffy = y_down - y_curr;
      dragging = (Math.abs(diffx) + Math.abs(diffy) > 10);
    }
  }, false);

  window.bullets = load_sprites({
    "BULLET_CREATE" : [[7], [0]],
    "BULLET_CREATE1" : [[7], [0]],
    "BULLET_CREATE2" : [[7], [0]],
    "BULLET_READY" : [[7], [0]],
    "BULLET_EXPLODE1" : [[0], [0]],
    "BULLET_EXPLODE2" : [[1], [0]],
    "BULLET_EXPLODE3": [[2], [0]],
    "_file" : "imgs/tiles.png",
    "_sizes" : [32, 32]
  });

  window.player_sprites = {
    "blue" : load_player_sprites("blue"),
    "red"  : load_player_sprites("red")
  };

  window.terrain_sprites = {};

  window.terrain_sprites["SOIL"] = load_sprites({
    "_file" : "rts/terrain/ground.png"
  });

  window.terrain_sprites["SAND"] = load_sprites({
    "_file" : "rts/terrain/sand.png"
  });

  window.terrain_sprites["GRASS"] = load_sprites({
    "_file" : "rts/terrain/grass.png"
  });

  window.terrain_sprites["ROCK"] = load_sprites({
    "_file" : "rts/terrain/rock.png"
  });

  window.terrain_sprites["WATER"] = load_sprites({
    "_file" : "rts/terrain/water.png"

  });

  window.terrain_sprites["FOG"] = load_sprites({
    "_file" : "rts/terrain/fog.png"
  });
};

var main = function () {
  var param = new URLSearchParams(window.location.search);
  if (param.has("player_type")) {
    var player_type =  param.get("player_type");
  } else {
    var player_type = "player";
  }
  if (param.has("port")) {
    var port = param.get("port");
  } else {
    var port = "8000";
  }
  if (param.has("mturk")) {
    var mturk = true;
  } else {
    var mturk = false;
  }
  start_game(player_type, port, mturk);
};


var has_unit_type = function(game, req_unit_type) {
  for (var i in game.units) {
    var unit = game.units[i];
    if (unit.temporary) {
      continue;
    }
    if (unit.unit_type == req_unit_type) {
      return true;
    }
  }
  return false;
}

var update_table = function(game, table, units, resource, is_unit) {
  for (var i in units) {
    var unit_name = units[i];
    var price = game.gamedef.units[window.unit_id[unit_name]].price;
    var build_from = game.gamedef.units[window.unit_id[unit_name]].build_from;

    if (price > resource) {
      table.rows[parseInt(i) + 1].cells[2].innerHTML = "<font color='red'><b>" + price + "</b></font>";
    } else {
      table.rows[parseInt(i) + 1].cells[2].innerHTML = "<font color='green'><b>" + price + "</b></font>";
    }

    if (is_unit) {
      if (has_unit_type(game, build_from)) {
        table.rows[parseInt(i) + 1].cells[3].innerHTML = "<font color='green'><b>Yes</b></font>";
      } else {
        table.rows[parseInt(i) + 1].cells[3].innerHTML = "<font color='red'><b>No</b></font>";
      }
    }
  }

}

var update_units_table = function(game, resource) {
  var table = document.getElementById("units_table");
  if (table === null || table.rows.length == 0) {
    return;
  }
  var units = ["PEASANT", "SWORDMAN", "SPEARMAN", "CAVALRY", "ARCHER", "DRAGON", "CATAPULT"];
  update_table(game, table, units, resource, true);
}

var update_buildings_table = function(game, resource) {
  var table = document.getElementById("buildings_table");
  if (table === null || table.rows.length == 0) {
    return;
  }
  var buildings = ["BARRACK", "BLACKSMITH", "STABLE", "WORKSHOP", "GUARD_TOWER", "TOWN_HALL"];
  update_table(game, table, buildings, resource, false);
}

var fill_table = function(game, units, table, with_produce) {
  var player_color = window.player_colors[game.player_id];
  var sprites = window.player_sprites[player_color];
  for (var i in units) {
    var unit_name = units[i];
    var tr = document.createElement("tr");

    var td_icon = document.createElement("td");
    var img = document.createElement("img");
    img.src = sprites[unit_name]._file;
    img.height = "28";
    img.width = "28";
    td_icon.appendChild(img);

    var td_name = document.createElement("td");
    td_name.appendChild(document.createTextNode(unit_name));

    var td_price = document.createElement("td");
    var price = game.gamedef.units[window.unit_id[unit_name]].price;
    td_price.appendChild(document.createTextNode(price));

    tr.appendChild(td_icon);
    tr.appendChild(td_name);
    tr.appendChild(td_price);

    if (with_produce) {
      var td_produce = document.createElement("td");
      var produce_table = document.createElement("table");
      var produce_tr = document.createElement("tr");
      var def = game.gamedef.units[window.unit_id[unit_name]];
      for (var j in def.build_skills) {
        var skill = def.build_skills[j];
        var skill_unit_name = window.unit_names_minirts[skill.unit_type];
        var skill_img = document.createElement("img");
        skill_img.src = sprites[skill_unit_name]._file;
        skill_img.height = "28";
        skill_img.width = "28";

        var td = document.createElement("td");
        td.style = "border: 0px";
        td.appendChild(skill_img);
        produce_tr.appendChild(td);
      }
      produce_table.appendChild(produce_tr);
      td_produce.appendChild(produce_table);

      tr.appendChild(td_produce);
    } else {
      var td_avl = document.createElement("td");
      td_avl.appendChild(document.createTextNode("Yes"));
      tr.appendChild(td_avl);
    }

    table.appendChild(tr);
  }
}

var fill_units_table = function(game) {
  var table = document.getElementById("units_table");
  if (table === null) {
    return;
  }
  var units = ["PEASANT", "SWORDMAN", "SPEARMAN", "CAVALRY", "ARCHER", "DRAGON", "CATAPULT"];
  fill_table(game, units, table, false);
}

var fill_buildings_table = function(game) {
  var table = document.getElementById("buildings_table");
  if (table === null) {
    return;
  }
  var buildings = ["BARRACK", "BLACKSMITH", "STABLE", "WORKSHOP", "GUARD_TOWER", "TOWN_HALL"];
  fill_table(game, buildings, table, true);
}

var show_interface = function() {
  var game_intf = document.getElementById("game_intf");
  if (game_intf != null) {
    game_intf.style.display = "block";
    var wait_partner = document.getElementById("wait_partner");
    if (wait_partner != null) {
      wait_partner.style.display = "none";
      swal("Your partner is ready now, the game is on!", "", "warning");
    }
  }


  var intf = document.getElementById("response-type-intf");
  if (intf != null) {
    intf.style.display = "none";
  }

  if (is_coach_intf()) {
    make_coach_intf();
  } else if (is_player_intf()) {
    make_player_intf();
  } else {
    make_spectator_intf();
  }
}

var start_game = function(player_type, port, mturk) {
  init();
  window.start_time = new Date();
  window.player_type = player_type;
  window.port = port;
  window.mturk = mturk;
  while (true) {
    try {
      if (mturk) {
        window.dealer = new WebSocket('wss://' + window.host + ':/wss' + window.port);
      } else {
        window.dealer = new WebSocket('ws://localhost:' + window.port);
      }
      break;
    } catch (error) {
      console.log(error);
    }
  }
  window.dealer.onopen = function(e) {
    console.log("WS Opened.");
  }

  window.onresize = function(e) {
    resize();
  }

  window.onload = function(e) {
    resize();
  }

  window.dealer.onmessage = function (message) {
    var s = message.data;
    var game = JSON.parse(s);
    window.last_state = game;
    window.map_x = game.rts_map.width;
    window.map_y = game.rts_map.height;
    window.ctx.clearRect(0, 0, window.canvas.width, window.canvas.height);
    render(game);
  };
};

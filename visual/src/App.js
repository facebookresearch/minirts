/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
import React, { Component } from 'react';
import './App.css';
import MapCanvas from "./MapCanvas";
import { loadData } from "./Util";
import Dropdown from 'react-dropdown';
import 'react-dropdown/style.css';

import ReactAutoComplete from "react-autocomplete";

const MAX_ITEMS = 50;

class App extends Component {

    // Keys:
    // "cons_count", "enemy_units", "instruction", "map",
    // "my_units", "resource", "resource_units", "tick", "unique_id",
    // "glob_cont", "base_frame_idx", "pre_ins_base_frame_idx", "prev_instruction"

    _assets = {};
    _games = [];
    _gamesFiltered = [];


    constructor(props) {
        super(props);
        this.state = {
            currFrame: 0,
            caption: "--",
            prevCaption: "--",
            gameIndex: -1,
            gameID: "",
            game: [],
            playing: false,
            loading: false,
            speed: 150,
            speedLabel: "Normal",
            targetTick: "",
            history: [],
            developerMode: true
        }

        // load all files
        loadData("./files.json").then((fs, i) => {
            for (let i = 0, len = fs.length; i < len; i++) {
                this._games.push({ index: i, label: fs[i], value: fs[i] });
            }
            this._gamesFiltered = this._games.slice(0, MAX_ITEMS);
            this.setState({ gameIndex: 0 });
        });

    }

    componentDidMount() {

    }

    // Load a specific game from the list
    handleSelectGame(val) {
        if (val !== this.state.gameID) {
            this.setState({ gameID: val, loading: true, playing: false, history: [], prevCaption: "--", caption: "--" });
            loadData(`${val}`).then((d) => {
                let game = [];
                for (let i = 0, len = d.length; i < len; i++) {
                    if (d[i] !== null) game.push(d[i]); // remove nulls
                }
                this.setState({ game: game, loading: false });
            });
        }
    }

    // Go to a specifc tick
    handleTickSearch(evt) {
        if (evt.which === 13) { // enter
            let g = this.state.game;
            if (g) {
                if (this.state.targetTick.indexOf("tick") !== 0) { // by frame number
                    let idx = parseInt(this.state.targetTick);
                    if (this.state.game && this.state.game[idx]) {
                        this.setState({ currFrame: idx, targetTick: "", playing: false });
                        return;
                    }

                } else { // by tick label
                    for (let i = 0, len = g.length; i < len; i++) {
                        if (g[i].tick === this.state.targetTick) {
                            this.setState({ currFrame: i, targetTick: "", playing: false });
                            return;
                        }
                    }
                }
            }
        }
    }


    updateGameInfo(t, p, f) {
        if (t !== this.state.caption) {
            let hist = this.state.history;
            hist.unshift({ frame: f, caption: `${hist.length + 1}. ${this.state.caption}` });
            this.setState({ prevCaption: this.state.caption });
        }
        this.setState({ caption: t, currFrame: f });

    }

    start() {
        this.setState({ playing: true, targetTick: "" });
    }

    stop() {
        this.setState({ playing: false });
    }

    forward() {
        this.setState({ currFrame: this.state.currFrame + 1, playing: false });
    }

    backward() {
        this.setState({ currFrame: Math.max(0, this.state.currFrame - 1), playing: false });
    }

    render() {

        let curr = this.state.game && this.state.game[this.state.currFrame] ? this.state.game[this.state.currFrame] : undefined;
        let hasGame = this.state.game.length > 0;

        return (
            <div className={"App" + (this.state.developerMode ? "" : " present")}>

                <div>
                    <div className="bg-title" onClick={e => this.setState({ developerMode: !this.state.developerMode })}>
                        MINI <br />
                        RTS
                    </div>

                    <div className="autocomplete">
                        <ReactAutoComplete
                            getItemValue={(item) => item.label}
                            items={this._gamesFiltered}
                            renderItem={(item, isHighlighted) =>
                                <div key={"i" + item.index} className="list-item" style={{ background: isHighlighted ? '#f1f3f7' : 'white' }}>
                                    {item.label}
                                </div>
                            }
                            menuStyle={{ zIndex: 1000, boxShadow: '0 2px 12px rgba(0, 0, 0, 0.1)', background: 'rgba(255, 255, 255, 0.9)', padding: '2px 0', fontSize: '90%', position: 'fixed', overflow: 'auto', maxHeight: '50%' }}
                            inputProps={{ placeholder: "Select game...", onMouseUp: (e) => document.querySelector("#file_input").select(), id: "file_input" }}
                            selectOnBlur={true}
                            value={this.state.gameID}
                            onChange={(e) => {
                                if (!e.target.value) {
                                    this._gamesFiltered = this._games.slice(0, MAX_ITEMS);
                                } else {
                                    this._gamesFiltered = this._games.filter(item => item.value.indexOf(e.target.value) >= 0).slice(0, MAX_ITEMS);
                                }
                                this.setState({ gameID: e.target.value });
                            }}
                            onSelect={this.handleSelectGame.bind(this)}
                            shouldItemRender={(item, val) => item.value.indexOf(this.state.gameID) >= 0}
                        />
                        {this.state.loading ? <span className="loading" role="img" aria-label="loading">⌛</span> : ""}
                    </div>

                    <div className="controls">
                        <div className={this.state.playing || !hasGame ? "hidden" : "btn play_btn"} onClick={this.start.bind(this)}>▶</div>
                        <div className={!this.state.playing || !hasGame ? "hidden" : "btn play_btn"} onClick={this.stop.bind(this)} >■</div>
                        <Dropdown className={"picker " + (!this.state.playing || !hasGame ? "hidden" : "")} onChange={(evt) => this.setState({ speed: evt.value, speedLabel: evt.label })} value={this.state.speedLabel}
                            options={[{ value: 50, label: "Fast" }, { value: 150, label: "Normal" }, { value: 300, label: "Slow" }]} />
                        <div className={"btn " + (!this.state.playing && hasGame ? "" : "hidden")} onClick={this.backward.bind(this)}>⇤</div>
                        <div className={hasGame ? "tickInputFrame" : "hidden"}><input className="tickInput" type="text" placeholder={curr ? curr.tick + " | " + this.state.currFrame : ""} value={this.state.targetTick}
                            onChange={e => this.setState({ targetTick: e.target.value })} onKeyUp={this.handleTickSearch.bind(this)} />
                        </div>
                        <div className={"btn " + (!this.state.playing && hasGame ? "" : "hidden")} onClick={this.forward.bind(this)}>⇥</div>
                    </div>
                </div>


                <div className="map-container">
                    <MapCanvas name="map" playing={this.state.playing}
                        game={this.state.game} index={this.state.gameID}
                        frame={this.state.currFrame} speed={this.state.speed}
                        gameCallback={this.updateGameInfo.bind(this)} background="#0af" />

                    <div className={"caption " + (hasGame ? "" : "hidden")}>
                        <h3>{this.state.caption}</h3>
                        <div className="caption_info">
                            Previous: {this.state.prevCaption} <br />
                        </div>
                    </div>
                </div>

                <div className="history">
                    {this.state.history.map((h, i) => i > 0 ? <div key={"h" + i}>{h.caption} ({h.frame})</div> : <div key="none" />)}
                </div>
            </div>
        );
    }
}

export default App;

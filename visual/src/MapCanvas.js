/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
import PtsCanvas from "react-pts-canvas";
import { loadAssets, getAsset } from './Util';
import Grid from "./Grid";
import { Pt, Rectangle } from "pts";


var lastTime = 0;


const CMD = {
    idle: 0, gather: 1, attack: 2, build_building: 3, build_unit: 4, move: 5, continue: 6
};

const CMD_COLORS = [ "rgba(255,255,255,.75)", "rgba(255,200,0, .75)", "rgba(255,0,50, .65)", "rgba(0,255,90, .75)", "rgba(0,100,255, .75)", "rgba(0,0,0,.3)", "rgba(255,255,255,.5)" ];


export default class MapCanvas extends PtsCanvas {


    constructor( props ) {
        super( props );

        loadAssets();
        this.curr = null;
        this.frame = 0;
        this.debug = false;

    }


    start( bound ) {
        this.grid = new Grid( this.space.innerBound, 32, 32 );
    }


    shouldComponentUpdate(nextProps, nextState) {
        if (this.props.index !== nextProps.index) {
            this.frame = 0;
            return true;
        }
        let update = (this.props.playing !== nextProps.playing || (!this.props.playing && this.frame !== nextProps.frame));
        if (update) this.frame = nextProps.frame;;
        return update;
    }



    animate( time, ftime ) {

        if (this.props.game && this.props.game.length > 0) {
            if (time - lastTime > this.props.speed) {
                lastTime = time;

                // play
                if (this.props.playing && this.frame < this.props.game.length-1) {
                    this.frame++;

                    // step
                } else if (!this.props.playing && this.props.frame !== this.frame) {
                    let diff = this.frame - this.props.frame;
                    this.frame = Math.max( 0, Math.min( this.props.game.length-1, this.frame - (diff/Math.abs(diff)) ));
                }

                this._drawCaption( this.props.game[this.frame].instruction, this.props.game[this.frame].prev_instruction );
            }

            let fr = this.props.game[this.frame];
            this.curr = fr;


            this.drawTerrain(fr.map);
            this.drawResource(fr.resource_units);
            this.drawUnit(fr.my_units);
			this.drawUnit(fr.enemy_units, "red");
			this.drawAmount(fr);
            // this.form.fill("#f03").text( [20, 20], `Frame ${this.frame} of ${this.props.game.length} (${fr.tick})` );
            if (this.debug) this.form.strokeOnly("rgba(255,255,255,.3)", 1).rects( this.grid.all );
        }
    }


    goTo( cmd, isEnemy ) {
        if (cmd.cmd_type === CMD.gather) {
            let r = this.curr.resource_units[ cmd.target_gather_idx ];
            if (r) return new Pt(r);
        } else if (cmd.cmd_type === CMD.attack ) {
            let enemy = isEnemy ? this.curr.my_units : this.curr.enemy_units;
            let r = enemy[ cmd.target_attack_idx ];
            if (r) return new Pt(r);
        } else if (cmd.cmd_type >= CMD.move ) {
            return new Pt(cmd.target_x, cmd.target_y);
        }
        return null;
    }


    drawTerrain( map ) {
        let terrain = map.terrain;
        let vis = map.visibility;
        if (!terrain || !vis) return;

        for (let i=0, len=terrain.length; i<len; i++) {
            this.form.image( getAsset("terrain", terrain[i]), this.grid.index(i) );
        }

        for (let i=0, len=vis.length; i<len; i++) {
            if (vis[i] !== 0) {
                this.form.fillOnly( "rgba(0,0,0,.3)" ).rect( this.grid.index(i) );
            }
        }
    }


    drawResource( items, side="blue" ) {
        if (!items) return;
        for (let i=0, len=items.length; i<len; i++) {
            this.form.image( getAsset(side, items[i].unit_type), this._rect( items[i] ) );
        }
    }


    drawUnit( items, side="blue" ) {
        if (!items) return;

        for (let i=0, len=items.length; i<len; i++) {
            let rect = this._rect( items[i] );
            this.drawCmd( items[i], side, rect );
            this.drawHP( items[i], rect.p1 );
        }
    }

    drawAmount( frame ) {
		let str = "$"+frame.resource;
		let m = this.space.ctx.measureText( str ) || 20;
		this.form.fillOnly("rgba(0,120,255,.85").rect( Rectangle.fromTopLeft( [0,0], [10+m.width+10, 30] ) );
        this.form.fillOnly("#fff").font(14, "bold").text( [10, 20], "$"+frame.resource );
    }


    _rect( unit ) {
        let ix = Math.floor( unit.x );
        let iy = Math.floor( unit.y );
        let pos = this.grid.at( ix, iy ).clone();
        pos.add(
            (unit.x-ix) * this.grid.cellsize - this.grid.cellsize/2,
            (unit.y-iy) * this.grid.cellsize - this.grid.cellsize/2
        );
        return pos;
    }

    drawHP( unit, pos ) {
        let half = this.grid.cellsize/2;
        let p1 = pos.$add( half/2, half*2+3 );
        this.form.strokeOnly("#f00", 3).line( [p1, p1.$add( half, 0 )] );
        this.form.strokeOnly("#0f0", 3).line( [p1, p1.$add( half * unit.hp, 0 )] );
    }


    drawCmd( unit, side, rect ) {

        // draw unit
        this.form.image( getAsset(side, unit.unit_type), rect );
        let cmd = unit.current_cmd;
        this._cmd( Rectangle.center( rect ), cmd, 1, side );

        // let tcmd = unit.target_cmd
        // this._cmd( pos, tcmd, 1, side );
        // this.form.fillOnly("rgba(0,0,0,.6)").text( pos.$subtract(this.grid.cellsize/4), cmd.cmd_type+":"+(tcmd ? tcmd.cmd_type : "") );
    }


    _cmd( pos, cmd, weight=1, side ) {
        let isEnemy = (side !== "blue");

        if (cmd && cmd.cmd_type > 0 && cmd.cmd_type < 6) {
            let target = this.goTo( cmd, isEnemy );
            if (target) {
                let tile = Rectangle.center( this._rect( target ) );
                this.form.strokeOnly( CMD_COLORS[ cmd.cmd_type ], weight ).line( [pos, tile] );
            }
        }

        // building in progress
        if (cmd && cmd.cmd_type === CMD.build_building) {
            this.space.ctx.save();
            this.space.ctx.globalAlpha = 0.3;
            this.form.image( getAsset(side, cmd.target_type), this._rect( new Pt(cmd.target_x, cmd.target_y) ) );
            this.space.ctx.restore();
        }
    }


    _drawCaption( t , p) {
        if (this.props.gameCallback) this.props.gameCallback(t, p, this.frame);
    }
}

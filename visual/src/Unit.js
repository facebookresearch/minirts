/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
import {Pt} from "pts";
import { getAsset } from './Util';

export const CMD = {
    idle: 0, gather: 1, attack: 2, build_building: 3, build_unit: 4, move: 5, continue: 6,
    labels: ["idle", "gather", "attack", "build_building", "build_unit", "move", "continue"]
}

export const CMD_COLORS = [ "#FFF", "#FE3", "#F03", "#0F6", "#09C", "rgba(0,0,0,.3)", "rgba(255,255,255,.5)" ];


export class Unit {


    constructor( unit, mapFn ) {
        this.state = {}; // readonly ref to current game state (don't modify)
        this.gridAt = mapFn;
        this.positionAt = (p, offset) => this.gridAt( p )[0].$add( offset );

        this.id = unit.unit_id;
        this.idx = unit.idx;
        this.type = unit.unit_type;
        this.hp = unit.hp;

        this.side = "blue";


        this.currPos = new Pt();
        this.nextPos = new Pt();

        this.cmdPos = new Pt();
        this.tarPos = new Pt();

        this.cmd = CMD.idle;
        this.tcmd = CMD.idle;

        this.offset = 0.25;
    }


    get resource() { return this.state.resource_units; }
    get team() { return (this.side === "blue") ? this.state.my_units : this.state.enemy_units; }
    get enemy() { return (this.side === "blue") ? tthis.state.enemy_units : this.state.my_units; }


    update( frame ) {
        this.state = frame;
        let self = (this.side === "blue") ? this.state.my_units[ this.idx ] : this.state.enemy_units[ this.idx ];
        this.cmd = self.current_cmd;
        this.tcmd = self.target_cmd;

        this.currPos = new Pt( this.nextPos );
        this.nextPos = this.positionAt( self, this.offset );
        this.cmdPos = (this.cmd) ? this.positionAt( this.cmd, this.offset ) : false;
        this.tarPos = (this.tcmd) ? this.positionAt( this.tcmd.x, this.offset ) : false;
    }


    draw( form ) {
        form.image( getAsset( this.side, this.type), this.gridAt( this.currPos ) );
        form.fillOnly("rgba(0,0,0,.6)").text( this.currPos, this.cmd.cmd_type+":"+(tcmd ? this.tcmd.cmd_type : "") );
        if (this.cmdPos) form.strokeOnly( CMD_COLORS[ this.cmd.cmd_type ], 1 ).line( [currPos, cmdPos] );
        if (this.tarPos) form.strokeOnly( CMD_COLORS[ this.tcmd.cmd_type ], 3 ).line( [currPos, tarPos] );
    }


    _findNext( cmd ) {

        // gather from resource location
        if ( cmd.cmd_type === CMD.gather ) {
            let r = this.resource[ cmd.target_gather_idx ];
            if (r) return this.positionAt( r.x, r.y, this.offset );

            // attack position
        } else if ( cmd.cmd_type === CMD.attack ) {
            let opp = (this.side === "blue") ? this.enemy : this.team;
            let r = opp[ cmd.target_attack_idx ];
            if (r) return this.positionAt( r.x, r.y, this.offset );

            // move position
        } else if ( cmd.cmd_type >= CMD.move ) {
            return this.positionAt( cmd.target_x, cmd.target_y, this.offset );
        }
        return false;
    }



}

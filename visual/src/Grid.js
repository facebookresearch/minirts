/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
import { Create, Rectangle } from "pts";


export default class Grid {

    constructor( bound, rows, cols ) {
        this._grid = Create.gridCells( bound, rows, cols );
        this._rows = rows;
        this._cols = cols;
        this._cell = Rectangle.size( this._grid[0] );
    }


    get all() { return this._grid; }
    get rows() { return this._rows; }
    get cols() { return this._cols; }
    get cellsize() { return this._cell.x; }
    get length() { return this._grid.length; }


    index( i ) {
        return this._grid[i];
    }


    at( x, y ) {
        return this.index( y * this._cols + x );
    }

    center( x, y ) {
        let c = (y === undefined) ? this.index(x) : this.at(x, y);
        return Rectangle.center( c );
    }

    offset( x, y, off ) {
        let c = (y === undefined) ? this.index(x) : this.at(x, y);
        return c[0].$add( off );
    }
}

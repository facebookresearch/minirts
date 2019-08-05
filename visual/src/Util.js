/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
export function loadData(url, method="GET", json=true) {
    return new Promise(function (resolve, reject) {
        var xhr = new XMLHttpRequest();
        xhr.open(method, url);
        xhr.onload = function () {
            console.log(xhr)
            if (this.status >= 200 && this.status < 300) {
                let c = json ? JSON.parse(xhr.response) : xhr.response;
                resolve(c);
            } else {
                reject({
                    status: this.status,
                    statusText: xhr.statusText
                });
            }
        };
        xhr.onerror = function () {
            reject({
                status: this.status,
                statusText: xhr.statusText
            });
        };
        xhr.send();
    });
}


/*
  custom_enum(
  Terrain,
  INVALID_TERRAIN = -1,
  SOIL = 0,
  SAND,
  GRASS,
  ROCK,
  WATER,
  FOG,
  NUM_TERRAIN);
*/

/*
  custom_enum(
  UnitType,
  INVALID_UNITTYPE = -1,
  // Minirts unit types
  RESOURCE = 0,
  PEASANT = 1,
  SPEARMAN = 2,
  SWORDMAN = 3,
  CAVALRY = 4,
  DRAGON = 5,
  ARCHER = 6,
  CATAPULT = 7,
  // buildings
  BARRACK = 8,
  BLACKSMITH = 9,
  STABLE = 10,
  WORKSHOP = 11,
  AVIARY = 12,
  ARCHERY = 13,
  GUARD_TOWER = 14,
  TOWN_HALL = 15,
  NUM_MINIRTS_UNITTYPE);
*/

var assets = {};
var path = "./assets/";

export function  loadAssets() {
    loadTerrain();
    loadUnit( "blue" );
    loadUnit( "red" );
}

export function loadTerrain() {
    let tns = ["soil", "sand", "grass", "rock", "water", "fog", "na"];
    assets["terrain"] = tns.map( t => _image( `${path}terrain/${t}.png` ) );
}

export function loadUnit(side="blue", style="medieval") {
    let units = [
        "resource", "peasant", "spearman", "swordman", "cavalry", "dragon", "archer", "catapult",
        "barrack", "blacksmith", "stable", "workshop", "aviary", "archery", "guard_tower", "town_hall", "na",
        "barrack_build", "blacksmith_build", "stable_build", "workshop_build", "aviary_build", "archery_build", "guard_tower_build", "town_hall_build",
    ];
    assets[`${side}`] = units.map( t => _image( `${path}${style}/${side}/${t}.png` ) );
}

export function getAsset( type, id, beingBuild ) {
    if (type === "terrain" && (isNaN(id) || id<0 || id>5)) {
        id = assets.terrain.length-1; // unknown terrain image, show "na.png"
    }

    if ( (type === "red" || type === "blue") && (isNaN(id) || id<0 || id>15)) {
        id = assets[type].length-1; // unknown terrain image, show "na.png"
    }

    if (beingBuild) {
        console.log( id, assets[type][id], assets[type][id+9] );
    }

    return assets[type][ id+(beingBuild ? 9 : 0) ];
}

function _image( src ) {
    var img = new Image();
    img.src = src;
    return img;
}

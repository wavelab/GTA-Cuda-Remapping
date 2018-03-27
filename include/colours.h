#pragma once

#define RGB(rv, gv, bv) ((((rv) & 0xFF) << 16) | (((gv) & 0xFF) << 8) | ((bv) & 0xFF))
#define GET_R(rv) (((rv) >> 16) & 0xFF)
#define GET_G(gv) (((gv) >> 8) & 0xFF)
#define GET_B(bv) ((bv) & 0xFF)

// #define cs_unlabeled  RGB(0,0,0)
// #define cs_ego_vehicle  RGB(0,0,0)
// #define cs_rectification_border  RGB(0,0,0)
// #define cs_out_of_roi  RGB(0,0,0)
// #define cs_static  RGB(0,0,0)
// #define cs_dynamic  RGB(111,74,0)
// #define cs_ground  RGB(81,0,81)
// #define cs_road  RGB(128,64,128)
// #define cs_sidewalk  RGB(244,35,232)
// #define cs_parking  RGB(250,170,160)
// #define cs_rail_track  RGB(230,150,140)
// #define cs_building  RGB(70,70,70)
// #define cs_wall  RGB(102,102,156)
// #define cs_fence  RGB(190,153,153)
// #define cs_guard_rail  RGB(180,165,180)
// #define cs_bridge  RGB(150,100,100)
// #define cs_tunnel  RGB(150,120,90)
// #define cs_pole  RGB(153,153,153)
// #define cs_polegroup  RGB(153,153,153)
// #define cs_traffic_light  RGB(250,170,30)
// #define cs_traffic_sign  RGB(220,220,0)
// #define cs_vegetation  RGB(107,142,35)
// #define cs_terrain  RGB(152,251,152)
// #define cs_sky  RGB(70,130,180)
// #define cs_person  RGB(220,20,60)
// #define cs_rider  RGB(255,0,0)
// #define cs_car  RGB(0,0,142)
// #define cs_truck  RGB(0,0,70)
// #define cs_bus  RGB(0,60,100)
// #define cs_caravan  RGB(0,0,90)
// #define cs_trailer  RGB(0,0,110)
// #define cs_train  RGB(0,80,100)
// #define cs_motorcycle  RGB(0,0,230)
// #define cs_bicycle  RGB(119,11,32)

// #define Dont_care_old RGB(0,0,0)
// #define bridge_old RGB(0,0,255)
// #define building_old RGB(0,255,0)
// #define construction_barrel_old RGB(0,255,255)
// #define construction_barricade_old RGB(255,0,0)
// #define crosswalk_old RGB(255,0,255)
// #define curb_old RGB(255,255,0)
// #define white_old RGB(255,255,255)
// #define debris_old RGB(41,41,41)
// #define fence_old RGB(156,41,41)
// #define guard_rail_old RGB(99,99,41)
// #define lane_separator_old RGB(214,99,41)
// #define pavement_marking_old RGB(41,156,41)
// #define rail_track_old RGB(156,156,41)
// #define road_old RGB(99,214,41)
// #define roadside_structure_old RGB(214,214,41)
// #define rumble_strip_old RGB(99,41,99)
// #define sidewalk_old RGB(214,41,99)
// #define terrain_old RGB(41,99,99)
// #define traffic_cone_old RGB(156,99,99)
// #define traffic_light_old RGB(99,156,99)
// #define traffic_marker_old RGB(214,156,99)
// #define traffic_sign_old RGB(41,214,99)
// #define tunnel_old RGB(156,214,99)
// #define utility_pole_old RGB(41,41,156)
// #define vegetation_old RGB(156,41,156)
// #define wall_old RGB(99,99,156)
// #define Car_old RGB(214,99,156)
// #define Trailer_old RGB(41,156,156)
// #define Bus_old RGB(156,156,156)
// #define Truck_old RGB(99,214,156)
// #define Airplane_old RGB(214,214,156)
// #define Moterbike_old RGB(99,41,214)
// #define Bycicle_old RGB(41,99,214)
// #define Boat_old RGB(156,99,214)
// #define Railed_old RGB(99,156,214)
// #define Pedestrian_old RGB(214,156,214)
// #define Animal_old RGB(156,214,214)

// #define Dont_care_new cs_unlabeled
// #define bridge_new cs_unlabeled
// #define building_new cs_building
// #define construction_barrel_new cs_unlabeled
// #define construction_barricade_new cs_unlabeled
// #define crosswalk_new cs_road
// #define curb_new cs_sidewalk
// #define white_new cs_unlabeled
// #define debris_new cs_unlabeled
// #define fence_new cs_fence
// #define guard_rail_new cs_wall
// #define lane_separator_new cs_road
// #define pavement_marking_new cs_road
// #define rail_track_new cs_unlabeled
// #define road_new cs_road
// #define roadside_structure_new cs_unlabeled
// #define rumble_strip_new cs_road
// #define sidewalk_new cs_sidewalk
// #define terrain_new cs_terrain
// #define traffic_cone_new cs_unlabeled
// #define traffic_light_new cs_traffic_light
// #define traffic_marker_new cs_unlabeled
// #define traffic_sign_new cs_traffic_sign
// #define tunnel_new cs_wall
// #define utility_pole_new cs_pole
// #define vegetation_new cs_vegetation
// #define wall_new cs_wall
// #define Car_new cs_car
// #define Trailer_new cs_truck
// #define Bus_new cs_bus
// #define Truck_new cs_truck
// #define Airplane_new cs_unlabeled
// #define Moterbike_new cs_motorcycle
// #define Bycicle_new cs_bicycle
// #define Boat_new cs_unlabeled
// #define Railed_new cs_unlabeled
// #define Pedestrian_new cs_person
// #define Animal_new cs_unlabeled

const unsigned int cs_road = 0;
const unsigned int cs_sidewalk = 1;
const unsigned int cs_building = 2;
const unsigned int cs_wall = 3;
const unsigned int cs_fence = 4;
const unsigned int cs_pole = 5;
const unsigned int cs_traffic_light = 6;
const unsigned int cs_traffic_sign = 7;
const unsigned int cs_vegetation = 8;
const unsigned int cs_terrain = 9;
const unsigned int cs_sky = 10;
const unsigned int cs_person = 11;
const unsigned int cs_rider = 12;
const unsigned int cs_car = 13;
const unsigned int cs_truck = 14;
const unsigned int cs_bus = 15;
const unsigned int cs_train = 16;
const unsigned int cs_motorcycle = 17;
const unsigned int cs_bicycle = 18;
const unsigned int cs_unlabeled = 255;

const unsigned int Dont_care_old = RGB(0,0,0);
const unsigned int bridge_old = RGB(0,0,255);
const unsigned int building_old = RGB(0,255,0);
const unsigned int construction_barrel_old = RGB(0,255,255);
const unsigned int construction_barricade_old = RGB(255,0,0);
const unsigned int crosswalk_old = RGB(255,0,255);
const unsigned int curb_old = RGB(255,255,0);
const unsigned int white_old = RGB(255,255,255);
const unsigned int debris_old = RGB(41,41,41);
const unsigned int fence_old = RGB(156,41,41);
const unsigned int guard_rail_old = RGB(99,99,41);
const unsigned int lane_separator_old = RGB(214,99,41);
const unsigned int pavement_marking_old = RGB(41,156,41);
const unsigned int rail_track_old = RGB(156,156,41);
const unsigned int road_old = RGB(99,214,41);
const unsigned int roadside_structure_old = RGB(214,214,41);
const unsigned int rumble_strip_old = RGB(214,41,99);
const unsigned int sidewalk_old = RGB(41,99,99);
const unsigned int terrain_old = RGB(156,99,99);
const unsigned int traffic_cone_old = RGB(99,156,99);
const unsigned int traffic_light_old = RGB(214,156,99);
const unsigned int traffic_marker_old = RGB(41,214,99);
const unsigned int traffic_sign_old = RGB(156,214,99);
const unsigned int tunnel_old = RGB(41,41,156);
const unsigned int utility_pole_old = RGB(156,41,156);
const unsigned int vegetation_old = RGB(99,99,156);
const unsigned int wall_old = RGB(214,99,156);
const unsigned int Car_old = RGB(41,156,156);
const unsigned int Trailer_old = RGB(156,156,156);
const unsigned int Bus_old = RGB(99,214,156);
const unsigned int Truck_old = RGB(214,214,156);
const unsigned int Airplane_old = RGB(99,41,214);
const unsigned int Moterbike_old = RGB(41,99,214);
const unsigned int Bycicle_old = RGB(156,99,214);
const unsigned int Boat_old = RGB(99,156,214);
const unsigned int Railed_old = RGB(214,156,214);
const unsigned int Pedestrian_old = RGB(214,156,214);
const unsigned int Animal_old = RGB(156,214,214);

const unsigned int Dont_care_new = cs_unlabeled;
const unsigned int bridge_new = cs_unlabeled;
const unsigned int building_new = cs_building;
const unsigned int construction_barrel_new = cs_unlabeled;
const unsigned int construction_barricade_new = cs_unlabeled;
const unsigned int crosswalk_new = cs_road;
const unsigned int curb_new = cs_sidewalk;
const unsigned int white_new = cs_unlabeled;
const unsigned int debris_new = cs_unlabeled;
const unsigned int fence_new = cs_fence;
const unsigned int guard_rail_new = cs_wall;
const unsigned int lane_separator_new = cs_road;
const unsigned int pavement_marking_new = cs_road;
const unsigned int rail_track_new = cs_unlabeled;
const unsigned int road_new = cs_road;
const unsigned int roadside_structure_new = cs_unlabeled;
const unsigned int rumble_strip_new = cs_road;
const unsigned int sidewalk_new = cs_sidewalk;
const unsigned int terrain_new = cs_terrain;
const unsigned int traffic_cone_new = cs_unlabeled;
const unsigned int traffic_light_new = cs_traffic_light;
const unsigned int traffic_marker_new = cs_unlabeled;
const unsigned int traffic_sign_new = cs_traffic_sign;
const unsigned int tunnel_new = cs_wall;
const unsigned int utility_pole_new = cs_pole;
const unsigned int vegetation_new = cs_vegetation;
const unsigned int wall_new = cs_wall;
const unsigned int Car_new = cs_car;
const unsigned int Trailer_new = cs_truck;
const unsigned int Bus_new = cs_bus;
const unsigned int Truck_new = cs_truck;
const unsigned int Airplane_new = cs_unlabeled;
const unsigned int Moterbike_new = cs_motorcycle;
const unsigned int Bycicle_new = cs_bicycle;
const unsigned int Boat_new = cs_unlabeled;
const unsigned int Railed_new = cs_unlabeled;
const unsigned int Pedestrian_new = cs_person;
const unsigned int Animal_new = cs_unlabeled;
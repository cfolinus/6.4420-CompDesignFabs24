$fn = 100;
difference() {
    cube([10,5,5], center=true);
    rotate([90,0,0]) translate([0,-2.40,0]) cylinder(10,3.40,3.40, center=true);
}

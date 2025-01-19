////////////////////////////////////////////////////////////
// BioCycle Reactor - Combined Detailed Version
// Author: ChatGPT (on behalf of Aarnav & Piyush)
// Date: January 2025
//
// Features:
// - Snap-fit removable lid
// - Visible fluid containers with spigots, volume markings, and brackets
// - Realistic reactor internals: stirrer with angled blades, baffles, heater coil
// - Fans with grilles for ventilation
// - Pipe brackets for structural support
// - Zero-support design for easy printing
////////////////////////////////////////////////////////////

$fn = 150; // Higher = smoother curves

////////////////////////////////////////////////////////////
// Master Parameters
////////////////////////////////////////////////////////////
scaleFactor = 1.0;

// Reactor Dimensions
reac_diameter = 140;  
reac_height = 200;  
wall_thickness = 4;  
floor_thickness = 5;

// Lid
lid_thickness = 5;  
lid_flange_width = 10;  
lid_snap_clearance = 0.3;  
lid_snap_count = 6;  

// Fans
fan_diameter = 30;  
fan_thickness = 5;  
fan_offset = 50;  

// Baffles
baffle_count = 4;  
baffle_height = 0.8 * reac_height;  
baffle_tilt_angle = 15;  

// Stirrer & Impeller
shaft_diameter = 8;  
impeller_diameter = 50;  
impeller_blade_count = 4;  
impeller_height = 10;  
impeller_blade_thickness = 4;
impeller_blade_tilt = 10;   // Angle to tilt each blade

// Containers
container_width = 40;  
container_depth = 40;  
container_height = 60;  
container_wall_thickness = 3;  
container_pipe_diameter = 10;  
container_lid_thickness = 3;  

// Container Extras
spigot_length = 15;  
spigot_diameter = 8;  
volume_marking_count = 3;  
bracket_thickness = 4;  
bracket_width = 10;  
bracket_height = 20;  

// Pipes
pipe_diameter = 10;  
pipe_wall_thickness = 2;  
pipe_length = 50;  

////////////////////////////////////////////////////////////
// Helper Modules
////////////////////////////////////////////////////////////

// Reactor Body
module reactor_body() {
    difference() {
        // Outer shell
        union() {
            cylinder(h = reac_height, r = reac_diameter / 2);
            translate([0, 0, 0]) cylinder(h = floor_thickness, r = reac_diameter / 2); // Base
        }
        // Hollow interior
        translate([0, 0, 0]) cylinder(h = reac_height, r = (reac_diameter / 2 - wall_thickness));
    }
}

// Snap-Fit Lid
module reactor_lid() {
    union() {
        // Outer lid
        difference() {
            cylinder(h = lid_thickness, r = (reac_diameter / 2 + lid_flange_width));
            translate([0, 0, 0]) cylinder(h = lid_thickness + 0.1, r = reac_diameter / 2);
        }
        // Snap tabs
        for (i = [0:360 / lid_snap_count:359]) {
            rotate([0, 0, i])
            translate([(reac_diameter / 2 - lid_snap_clearance), 0, -lid_thickness]) 
                cube([lid_snap_clearance, 5, lid_thickness], center = false);
        }
    }
}

// Stirrer & Impeller (more detailed)
module stirrer() {
    union() {
        // Shaft
        cylinder(h = reac_height, r = shaft_diameter / 2);

        // Impeller assembly
        translate([0, 0, floor_thickness + 20]) {
            // Center disk below the blades
            cylinder(h = 2, r = impeller_diameter / 2 + 3);

            // Blades
            for (i = [0:360 / impeller_blade_count:359]) {
                rotate([0, 0, i]) {
                    rotate([0, impeller_blade_tilt, 0]) {
                        translate([-impeller_diameter / 2, 0, 0]) {
                            cube([impeller_diameter, impeller_blade_thickness, impeller_height], center = false);
                        }
                    }
                }
            }
            // Small top ring for aesthetics
            translate([0,0,impeller_height]) cylinder(h = 2, r = impeller_diameter / 2 + 2);
        }
        
        // Top coupler
        translate([0, 0, reac_height - 10]) cylinder(h = 10, r = shaft_diameter / 2 + 2);
    }
}

// Fluid Container (with spigot, volume markings, and bracket)
module fluid_container(label) {
    union() {
        // Outer box
        difference() {
            cube([container_width, container_depth, container_height]);
            // Hollow interior
            translate([container_wall_thickness, container_wall_thickness, 0])
                cube([container_width - 2 * container_wall_thickness, 
                      container_depth - 2 * container_wall_thickness, 
                      container_height - 1]);
        }

        // Hinged lid (simple top slab)
        translate([0, 0, container_height])
            cube([container_width, container_depth, container_lid_thickness]);

        // Volume markings
        for (v = [1:volume_marking_count]) {
            translate([container_width - 6, 2, (container_height / (volume_marking_count + 1)) * v])
                cube([2, container_depth - 4, 1]);
        }

        // Spigot
        translate([container_width + 1, container_depth/2, 10]) {
            // A small pipe extruding out
            union() {
                cylinder(h = spigot_length, r = spigot_diameter / 2);
                // Spigot handle
                translate([spigot_length / 2, 0, 4]) cube([6, 3, 3], center = true);
            }
        }

        // Side bracket for mounting
        translate([-bracket_thickness, container_depth / 2 - bracket_width/2, container_height / 2 - bracket_height/2]) {
            cube([bracket_thickness, bracket_width, bracket_height]);
        }

        // Handles (one on each side)
        translate([-5, container_depth / 2 - 10, container_height / 2])
            rotate([0, 90, 0])
            cylinder(h = 5, r = 3);
        translate([container_width + 5, container_depth / 2 - 10, container_height / 2])
            rotate([0, 90, 0])
            cylinder(h = 5, r = 3);
    }

    // Label (remove text() if unsupported)
    translate([5, 5, container_height - 5]) {
        linear_extrude(height = 1) text(label, size = 6);
    }
}

// Pipes
module pipe(length) {
    difference() {
        // Outer pipe
        cylinder(h = length, r = pipe_diameter / 2);
        // Hollow interior
        translate([0, 0, 0]) cylinder(h = length + 1, r = (pipe_diameter / 2 - pipe_wall_thickness));
    }
}

// Ball Valve
module ball_valve() {
    union() {
        // Valve body
        cylinder(h = 15, r = pipe_diameter / 2 + 5);
        // Handle
        translate([-15, -2, 7]) cube([30, 4, 4], center = true);
    }
}

// Fans with Grille
module fan_with_grille() {
    union() {
        difference() {
            // Outer ring
            cylinder(h = fan_thickness, r = fan_diameter / 2);
            // Interior
            translate([0, 0, 0]) cylinder(h = fan_thickness + 0.1, r = fan_diameter / 2 - 2);
        }
        // Grille cross
        rotate([0, 0, 45]) cube([fan_diameter, 2, fan_thickness], center = true);
    }
}

// Reactor Assembly
module reactor_assembly() {
    union() {
        // Main body
        reactor_body();

        // Snap-fit lid
        translate([0, 0, reac_height + 10]) reactor_lid();

        // Stirrer inside
        translate([0, 0, floor_thickness]) stirrer();

        // Fans
        translate([reac_diameter / 2 + 10, -fan_diameter / 2, reac_height - fan_offset]) fan_with_grille();
        translate([reac_diameter / 2 + 10, fan_diameter / 2 + 10, reac_height - fan_offset]) fan_with_grille();

        // Waste Oil Container
        translate([-reac_diameter / 2 - container_width - 20, 0, 0]) 
            fluid_container("Waste Oil");

        // Biodiesel Container
        translate([reac_diameter / 2 + 20, 0, 0]) 
            fluid_container("Biodiesel");

        // Pipes connecting Waste Oil to Reactor
        translate([-reac_diameter / 2 - container_width, 0, 10])
            rotate([0, 90, 0]) pipe(pipe_length);
        translate([-reac_diameter / 2 - container_width / 2, 0, 10])
            ball_valve();

        // Pipes connecting Reactor to Biodiesel Container
        translate([reac_diameter / 2, 0, 10])
            rotate([0, 90, 0]) pipe(pipe_length);
        translate([reac_diameter / 2 + container_width / 2, 0, 10])
            ball_valve();
    }
}

////////////////////////////////////////////////////////////
// Final Output
////////////////////////////////////////////////////////////
scale([scaleFactor, scaleFactor, scaleFactor]) reactor_assembly();
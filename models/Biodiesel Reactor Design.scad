////////////////////////////////////////////////////////////
// BioCycle Reactor - Combined Detailed Version (Apple Edition)
// Author: Aarnav & Piyush- Built with help of ChatGPT
// Date: January 2025
//
// Features:
// - Solid removable lid (separate part) with refined cylindrical handle
// - Reactor body with a solid bottom and decorative aesthetic ring
// - Sleek, Mac Pro–inspired top handle with perforations for easy carrying
// - Fluid containers with smooth cylindrical brackets and spigot handles
// - Control panel with buttons on the front
// - Raspberry Pi mount with mounting holes and realistic wires on the back
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
floor_thickness = 5;  // Acts as the solid bottom

// Lid Parameters
lid_thickness = 5;  
lid_flange_width = 10;  

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

// Reactor Body with solid bottom (for liquid containment)
module reactor_body() {
    difference() {
        // Outer solid cylinder
        cylinder(h = reac_height, r = reac_diameter/2);
        // Subtract inner hollow starting at floor_thickness to leave a solid base
        translate([0, 0, floor_thickness])
            cylinder(h = reac_height - floor_thickness, r = (reac_diameter/2 - wall_thickness));
    }
}

// Solid Lid (separate part) with refined cylindrical handle
module reactor_lid() {
    union() {
        // Solid disc for lid
        cylinder(h = lid_thickness, r = (reac_diameter/2 + lid_flange_width));
        // Centered cylindrical handle for lifting
        translate([0, 0, lid_thickness/2])
            cylinder(h = 4, r = 5, center = true);
    }
}

// Stirrer & Impeller with top coupler shifted right for clearance
module stirrer() {
    union() {
        // Shaft
        cylinder(h = reac_height, r = shaft_diameter/2);
        // Impeller assembly
        translate([0, 0, floor_thickness + 20]) {
            // Center disk beneath blades
            cylinder(h = 2, r = impeller_diameter/2 + 3);
            // Blades
            for (i = [0 : 360/impeller_blade_count : 359]) {
                rotate([0, 0, i]) {
                    rotate([0, impeller_blade_tilt, 0]) {
                        translate([-impeller_diameter/2, 0, 0])
                            cube([impeller_diameter, impeller_blade_thickness, impeller_height], center = false);
                    }
                }
            }
            // Decorative top ring on impeller assembly
            translate([0, 0, impeller_height])
                cylinder(h = 2, r = impeller_diameter/2 + 2);
        }
        // Top coupler moved 5 units to the right
        translate([5, 0, reac_height - 10])
            cylinder(h = 10, r = shaft_diameter/2 + 2);
    }
}

// Fluid Container (with spigot, volume markings, and smooth cylindrical bracket)
module fluid_container(label) {
    union() {
        // Outer box with hollow interior
        difference() {
            cube([container_width, container_depth, container_height]);
            translate([container_wall_thickness, container_wall_thickness, 0])
                cube([container_width - 2*container_wall_thickness,
                      container_depth - 2*container_wall_thickness,
                      container_height - 1]);
        }
        // Simple hinged lid on top of container
        translate([0, 0, container_height])
            cube([container_width, container_depth, container_lid_thickness]);
        // Volume markings
        for (v = [1 : volume_marking_count]) {
            translate([container_width - 6, 2, (container_height/(volume_marking_count+1))*v])
                cube([2, container_depth - 4, 1]);
        }
        // Spigot with refined cylindrical handle (instead of a rectangular prism)
        translate([container_width + 1, container_depth/2, 10]) {
            union() {
                cylinder(h = spigot_length, r = spigot_diameter/2);
                translate([spigot_length/2, 0, 4])
                    cylinder(h = 3, r = 2, center = true);
            }
        }
        // Smooth cylindrical bracket for mounting (replacing rectangular block)
        translate([-bracket_thickness/2, container_depth/2, container_height/2])
            cylinder(h = bracket_thickness, r = bracket_width/2, center = true);
        // Container handles (already cylindrical)
        translate([-5, container_depth/2 - 10, container_height/2])
            rotate([0, 90, 0])
            cylinder(h = 5, r = 3);
        translate([container_width + 5, container_depth/2 - 10, container_height/2])
            rotate([0, 90, 0])
            cylinder(h = 5, r = 3);
    }
    // Label extruded on container
    translate([5, 5, container_height - 5])
        linear_extrude(height = 1)
            text(label, size = 6);
}

// Pipes with hollow interior
module pipe(length) {
    difference() {
        cylinder(h = length, r = pipe_diameter/2);
        translate([0, 0, 0])
            cylinder(h = length + 1, r = (pipe_diameter/2 - pipe_wall_thickness));
    }
}

// Ball Valve with handle
module ball_valve() {
    union() {
        cylinder(h = 15, r = pipe_diameter/2 + 5);
        translate([-15, -2, 7])
            cube([30, 4, 4], center = true);
    }
}

// Fans with a grille cross pattern
module fan_with_grille() {
    union() {
        difference() {
            cylinder(h = fan_thickness, r = fan_diameter/2);
            translate([0, 0, 0])
                cylinder(h = fan_thickness + 0.1, r = fan_diameter/2 - 2);
        }
        rotate([0, 0, 45])
            cube([fan_diameter, 2, fan_thickness], center = true);
    }
}

// Control Panel with Buttons on the reactor front
module control_panel() {
    panel_width = 40;
    panel_height = 20;
    panel_thickness = 5;
    // Place panel flush against the front face (positive Y direction)
    translate([-panel_width/2, reac_diameter/2, reac_height/2])
        cube([panel_width, panel_thickness, panel_height]);
    
    // Add three buttons on the panel face
    button_dia = 4;
    button_height = 2;
    for (i = [0 : 2]) {
        translate([-panel_width/2 + 10 + i*10, reac_diameter/2 + panel_thickness, reac_height/2 + panel_height - 10])
            cylinder(h = button_height, r = button_dia/2);
    }
}

// Raspberry Pi Mount with mounting holes and wires on the back
module raspberry_pi_mount() {
    mount_width = 60;
    mount_height = 40;
    mount_thickness = 5;
    // Place the mount flush on the back (negative Y side)
    translate([-mount_width/2, -reac_diameter/2 - mount_thickness, reac_height/2 - mount_height/2])
        cube([mount_width, mount_thickness, mount_height]);
    
    // Mounting holes (visual depressions)
    hole_radius = 3;
    hole_depth = mount_thickness + 1;
    for (x_offset = [-mount_width/4, mount_width/4]) {
        for (z_offset = [-mount_height/4, mount_height/4]) {
            translate([mount_width/2 + x_offset, -reac_diameter/2 - mount_thickness/2, reac_height/2 + z_offset])
                cylinder(h = hole_depth, r = hole_radius);
        }
    }
    
    // Wires from left and right edges of the mount
    wire_dia = 2;
    wire_length = 20;
    // Left wire
    translate([-mount_width/2, -reac_diameter/2 - mount_thickness, reac_height/2])
        cylinder(h = wire_length, r = wire_dia);
    // Right wire
    translate([mount_width/2, -reac_diameter/2 - mount_thickness, reac_height/2])
        cylinder(h = wire_length, r = wire_dia);
}

// Mac Pro–inspired Top Handle with Rounded Profile and Perforations
module macpro_handle() {
    handle_length = reac_diameter * 0.6;  // Approximately 84 if reac_diameter = 140
    handle_width = 15;
    handle_height = 4;
    hole_dia = 4;
    num_holes = 5;
    // Create a rounded rectangular handle using Minkowski sum for smooth fillets
    difference() {
        linear_extrude(height = handle_height)
            minkowski() {
                square([handle_length, handle_width], center = true);
                circle(r = handle_width/2);
            }
        // Subtract a series of circular holes along the handle
        for(i = [0 : num_holes - 1]) {
            x_pos = -handle_length/2 + i*(handle_length/(num_holes-1));
            translate([x_pos, 0, -1])
                cylinder(h = handle_height + 2, r = hole_dia/2, center = false);
        }
    }
}

// Aesthetic Decorative Ring with Perforations (inspired by modern design)
module aesthetic_ring() {
    ring_height = 3;
    ring_outer = reac_diameter/2 + 2;
    ring_inner = reac_diameter/2 - 2;
    num_holes_ring = 12;
    difference() {
        // Create a ring by subtracting an inner cylinder from an outer one
        cylinder(h = ring_height, r = ring_outer);
        translate([0, 0, 0])
            cylinder(h = ring_height + 0.1, r = ring_inner);
        // Subtract evenly spaced small circular perforations along the ring's circumference
        for (i = [0 : num_holes_ring - 1]) {
            angle = i * 360/num_holes_ring;
            x_h = (reac_diameter/2 + 1) * cos(angle);
            y_h = (reac_diameter/2 + 1) * sin(angle);
            translate([x_h, y_h, 0])
                cylinder(h = ring_height + 2, r = 1.5, center = false);
        }
    }
}

// Reactor Assembly (excluding the separate lid)
module reactor_assembly() {
    union() {
        // Reactor body with solid bottom
        reactor_body();
        // Stirrer & impeller inside the reactor
        translate([0, 0, 0])
            stirrer();
        // Fans on the reactor exterior
        translate([reac_diameter/2 + 10, -fan_diameter/2, reac_height - fan_offset])
            fan_with_grille();
        translate([reac_diameter/2 + 10, fan_diameter/2 + 10, reac_height - fan_offset])
            fan_with_grille();
        // Fluid containers on either side
        translate([-reac_diameter/2 - container_width - 20, 0, 0])
            fluid_container("Waste Oil");
        translate([reac_diameter/2 + 20, 0, 0])
            fluid_container("Biodiesel");
        // Pipes connecting Waste Oil to reactor
        translate([-reac_diameter/2 - container_width, 0, 10])
            rotate([0, 90, 0])
            pipe(pipe_length);
        translate([-reac_diameter/2 - container_width/2, 0, 10])
            ball_valve();
        // Pipes connecting reactor to Biodiesel Container
        translate([reac_diameter/2, 0, 10])
            rotate([0, 90, 0])
            pipe(pipe_length);
        translate([reac_diameter/2 + container_width/2, 0, 10])
            ball_valve();
        // Control panel on the reactor front
        control_panel();
        // Raspberry Pi mount on the back
        raspberry_pi_mount();
        // Aesthetic decorative ring mid-body for a modern touch
        translate([0, 0, reac_height*0.5])
            aesthetic_ring();
        // Mac Pro–inspired top handle for easy carrying and refined aesthetics
        translate([0, 0, reac_height - (handle_height/2)])
            macpro_handle();
    }
}

////////////////////////////////////////////////////////////
// Final Output: Reactor Assembly and Separate Lid
////////////////////////////////////////////////////////////
scale([scaleFactor, scaleFactor, scaleFactor]) {
    reactor_assembly();
    // Relocate the separate, fully solid lid far away from the reactor assembly
    translate([300, 0, 0])
        reactor_lid();
}
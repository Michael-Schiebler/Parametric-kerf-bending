import ezdxf
import numpy as np
from ezdxf.math import Vec3, Vec2
from ezdxf.enums import TextEntityAlignment
from math import atan2, sqrt, pi,radians, tan, ceil, degrees
from scipy import interpolate
from scipy.interpolate import splprep, splev
import math

debug = False


#TO DO LIST 
#  X - fixed, splines were computed as vec3 list not numpy arrays - splines and lenght diff algo do not mix, probably a stupid fix
#in the lenght diff algo layer assignment barely works and seems to draw the wrong colors, the list displayed in html is fine though, probably indexing error
#I should stop using for i in range(len(list)
#arcs are weird, should steal from blocklayer

def find_nearest_index(point, curve_points):
    """Find the index of the nearest point in curve_points to the given point"""
    min_dist = float('inf')
    min_idx = -1
    
    for i, p in enumerate(curve_points):
        dist = (point[0] - p[0])**2 + (point[1] - p[1])**2  # Squared distance, no sqrt needed
        if dist < min_dist:
            min_dist = dist
            min_idx = i
            
    return min_idx

def offset_curve(curve_points, offset_distance):
    offset_points = []
    for i in range(len(curve_points)):
        if i == 0:
            p1 = curve_points[i]
            p2 = curve_points[i+1]
        elif i == len(curve_points)-1:
            p1 = curve_points[i-1]
            p2 = curve_points[i]
        else:
            p1 = curve_points[i-1]
            p2 = curve_points[i+1]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length == 0:
            normal = np.array([0, 0])
        else:
            normal = np.array([-dy, dx]) / length
        

        offset_pt = np.array([curve_points[i][0], curve_points[i][1]]) + offset_distance * normal
        #offset_pt = np.array(curve_points[i][:2]) + offset_distance * normal
        offset_points.append((offset_pt[0], offset_pt[1], 0))
    return offset_points

def has_double_curvature(control_points):
    # Need at least 3 points to form a curve (otherwise it's a straight line)
    if len(control_points) < 3:
        return False
    
    # First and last points (the line we compare against)
    x0, y0 = control_points[0][0], control_points[0][1]
    xn, yn = control_points[-1][0], control_points[-1][1]
    
    # Early exit if first and last points are the same (degenerate case)
    if x0 == xn and y0 == yn:
        return False
    
    # We'll track which side the points are on
    prev_side = None
    has_positive = False
    has_negative = False
    
    # Check all points except first and last
    for i in range(1, len(control_points) - 1):
        x, y = control_points[i][0], control_points[i][1]
        
        # Calculate which side of the line the point is on
        # Formula: (y - y0)(xn - x0) - (yn - y0)(x - x0)
        side = (y - y0) * (xn - x0) - (yn - y0) * (x - x0)
        
        # If side is exactly 0, the point is ON the line (skip)
        if side == 0:
            continue
        
        # Track if we've seen points on both sides
        if side > 0:
            has_positive = True
        else:
            has_negative = True
        
        # Early exit if we already have both sides
        if has_positive and has_negative:
            return True
    
    # If we have points on both sides, it's S-shaped
    return has_positive and has_negative

def ray_segment_intersection(ray_origin, angle, seg_start, seg_end, epsilon=1e-6):
    # Extract 2D coordinates
    x1, y1 = ray_origin[0], ray_origin[1]
    x3, y3 = seg_start[0], seg_start[1]
    x4, y4 = seg_end[0], seg_end[1]
    
    # Ray direction vector
    x2 = x1 + math.cos(angle) 
    y2 = y1 + math.sin(angle)
    
    # Calculate denominator
    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    # Lines are parallel if denominator is zero
    if abs(den) < epsilon:
        return None
    
    # Calculate parameters
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
    
    # Check if intersection occurs on the segment and in ray direction
    if ua >= 0 and 0 <= ub <= 1:
        # Calculate intersection point
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y, ray_origin[2])
    
    return None


def get_vector_between2p(points_list, id1, id2):
    x1, y1 = points_list[id1][0], points_list[id1][1]
    x2, y2 = points_list[id2][0], points_list[id2][1]
    dx = x2 - x1
    dy = y2 - y1
    return Vec2(dx, dy).normalize()

def remove_boundary_cuts(cut_points, side_list):
    """Remove boundary cuts - cuts that switch between side A and B"""
    if len(cut_points) <= 3:  # Not enough cuts to process
        return cut_points, side_list
    
    # Start with the original first and last points (which are 'Undefined')
    filtered_points = [cut_points[0]]
    filtered_sides = [side_list[0]]
    
    # Scan through the middle points (excluding first and last)
    for i in range(1, len(cut_points) - 1):
        # Check if this is a boundary cut
        if (i > 1 and i < len(cut_points) - 2 and  # Make sure we're not at the edges
            side_list[i] != 'Undefined' and        # Current cut is defined
            side_list[i-1] != 'Undefined' and      # Previous cut is defined
            side_list[i+1] != 'Undefined' and      # Next cut is defined
            side_list[i-1] == side_list[i] and     # Current matches previous
            side_list[i] != side_list[i+1]):       # But different from next
            # Skip this boundary cut
            continue
        
        # Otherwise keep the cut
        filtered_points.append(cut_points[i])
        filtered_sides.append(side_list[i])
    
    # Always include the last point
    filtered_points.append(cut_points[-1])
    filtered_sides.append(side_list[-1])
    
    return filtered_points, filtered_sides

def signed_angle_between(v1, v2):
    # Returns signed angle between v1 and v2 in radians
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    det = unit_v1[0]*unit_v2[1] - unit_v1[1]*unit_v2[0]  # 2D cross product
    angle = np.arctan2(det, dot)  # atan2 gives signed result
    return angle  # Range: [-π, π]

def estimate_curve_length(points, curve_type="spline", spline_tension=0.5, samples=20):
    """Estimate curve length with a rough approximation"""
    if curve_type == "arc":
        # For arc, we can directly calculate the length
        arc_radius = points[0]  # assuming arc_radius is passed as first element
        arc_angle = points[1]   # assuming arc_angle is passed as second element
        return radians(arc_angle) * arc_radius
    
    # For bezier or spline, generate a rough approximation with few samples
    if curve_type == "bezier" or curve_type == "spline":
        rough_points = create_curve_points(points, samples, curve_type, spline_tension)
        
        # Calculate the total length by summing segment lengths
        total_length = 0
        for i in range(len(rough_points) - 1):
            total_length += two_p_distance(rough_points[i], rough_points[i+1])
        
        return total_length
    
    return 0  # Default fallback

def create_curve_points(points, samples, curve_type="spline", spline_tension=0.5):
    """Create curve points based on curve type"""
    if len(points) == 3:
        # Quadratic bezier curve handling
        return create_bezier_points(points, samples)
    elif len(points) > 3 and curve_type == "bezier":
        # Higher-order bezier curve handling
        return create_higher_order_bezier_points(points, samples)
    elif len(points) > 3:
        # Spline curve handling
        return create_spline_points(points, samples, spline_tension)

def create_bezier_points(points, samples):
    """Create bezier curve points"""
    p0, p1, p2 = map(Vec3, [(p[0], p[1], 0) for p in points])
    curve_points = []

    for t in np.linspace(0, 1, samples + 1):
        # Calculate point on curve
        point = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
        curve_points.append(point)

    return curve_points

def create_spline_points(points, samples, spline_tension=0.5):
    """Create cubic spline curve points"""
    # Extract x and y coordinates
    x_points = np.array([p[0] for p in points])
    y_points = np.array([p[1] for p in points])

    # Create a parameter array based on cumulative chord length
    t = np.zeros(len(points))
    for i in range(1, len(points)):
        t[i] = t[i-1] + sqrt((x_points[i] - x_points[i-1])**2 + (y_points[i] - y_points[i-1])**2)

    # Normalize parameter values
    if t[-1] > 0:
        t = t / t[-1]

    # Create cubic spline interpolations with tension control
    smoothing_factor = (1 - spline_tension) * len(points)

    # Create spline for x and y coordinates
    tck_x = interpolate.splrep(t, x_points, s=smoothing_factor)
    tck_y = interpolate.splrep(t, y_points, s=smoothing_factor)

    # Generate points along the spline
    u = np.linspace(0, 1, samples + 1)
    x_spline = interpolate.splev(u, tck_x)
    y_spline = interpolate.splev(u, tck_y)

    # Create curve points
    curve_points = [Vec3(x, y, 0) for x, y in zip(x_spline, y_spline)]

    curve_points = np.array([[p.x, p.y, p.z] for p in curve_points])

    return curve_points

def create_higher_order_bezier_points(points, samples):
    """Create higher-order bezier curve points"""
    # Convert points to Vec3 objects for consistent handling
    control_points = [Vec3(p[0], p[1], 0) for p in points]
    curve_points = []
    
    # Generate points along the curve
    for t in np.linspace(0, 1, samples + 1):
        # Calculate point on curve using De Casteljau algorithm
        point = de_casteljau_point(control_points, t)
        curve_points.append(point)
    
    return curve_points

def de_casteljau_point(control_points, t):
    """Compute a point on a Bezier curve using the De Casteljau algorithm"""
    if len(control_points) == 1:
        return control_points[0]
    
    # Recursive step: compute points at level n-1
    new_points = []
    for i in range(len(control_points) - 1):
        # Linear interpolation
        new_point = (1-t) * control_points[i] + t * control_points[i+1]
        new_points.append(new_point)
    
    # Recursively compute until we get to a single point
    return de_casteljau_point(new_points, t)

def draw_curve(msp, points, color=0):
    """Draw a curve connecting the given points"""
    #for i in range(len(points)-1):      
    #    if color == 5: modelspace.add_line(points[i], points[i+1], dxfattribs={"layer": "Original Curve"})      #lazy way to handle layers but idc
    #    else: modelspace.add_line(points[i], points[i+1], dxfattribs={"color": color})
    if color == 5: msp.add_lwpolyline(points, dxfattribs={"layer": "Original Curve"})
    else: msp.add_lwpolyline(points, dxfattribs={"color": color})

def draw_line(msp,pointx, pointy,angle, length):
        direction = Vec2.from_angle(angle) * length
        start=Vec2(pointx, pointy, 0)
        msp.add_line(
            start,
            end = (start + direction),
           # dxfattribs={"layer": curve } 
        )
def two_p_distance(p1, p2):
    """Calculate distance between two points"""
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def sample_curve(entity, num_points=100):
    """Sample points along a curve entity."""
    try:
        curve = entity.to_curve3d()
        return np.array([curve.point(t) for t in np.linspace(0, 1, num_points)])
    except Exception as e:
        print(f"Could not sample curve: {e}")
        return np.empty((0, 3))



def import_dxf(msp, samples_per_entity=5):

    points = []

    for entity in msp:
        try:
            if entity.dxftype() == "POINT":
                location = entity.dxf.location
                points.append([location.x, location.y, location.z])

            elif entity.dxftype() == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                for t in np.linspace(0, 1, samples_per_entity):
                    x = start.x + t * (end.x - start.x)
                    y = start.y + t * (end.y - start.y)
                    z = start.z + t * (end.z - start.z)
                    points.append([x, y, z])

            elif entity.dxftype() in {"LWPOLYLINE", "POLYLINE"}:
                polyline_points = []
                for v in entity:
                    point = v.dxf.location if hasattr(v.dxf, "location") else (v.dxf.x, v.dxf.y, getattr(v.dxf, "z", 0.0))
                    polyline_points.append(point)

                for i in range(len(polyline_points) - 1):
                    p0 = polyline_points[i]
                    p1 = polyline_points[i+1]
                    for t in np.linspace(0, 1, samples_per_entity // (len(polyline_points) - 1)):
                        x = p0[0] + t * (p1[0] - p0[0])
                        y = p0[1] + t * (p1[1] - p0[1])
                        z = (p0[2] if len(p0) > 2 else 0.0) + t * ((p1[2] if len(p1) > 2 else 0.0) - (p0[2] if len(p0) > 2 else 0.0))
                        points.append([x, y, z])

            elif entity.dxftype() == "SPLINE":
                try:
                    fit_pts = np.array(entity.fit_points)
                    if len(fit_pts) < 2:
                        print("Skipping SPLINE with < 2 fit points")
                        continue

                    # Convert to x, y, z arrays
                    x, y, z = fit_pts.T
                    k = min(3, len(fit_pts) - 1)  # spline degree (must be < number of points)

                    # Fit parametric spline
                    tck, _ = splprep([x, y, z], k=k, s=0)

                    # Evaluate at N evenly spaced points
                    u_fine = np.linspace(0, 1, len(fit_pts)*250)    #fixed value due to laziness
                    x_new, y_new, z_new = splev(u_fine, tck)

                    for x_i, y_i, z_i in zip(x_new, y_new, z_new):
                        points.append([x_i, y_i, z_i])

                except Exception as e:
                    print(f"Skipping SPLINE (spline fit failed): {e}")


                except Exception as e:
                    print(f"Skipping SPLINE due to error: {e}")


            elif entity.dxftype() in {"CIRCLE", "ARC", "ELLIPSE"}:
                # Use .to_curve3d() where available
                try:
                    curve = entity.to_curve3d()
                    for t in np.linspace(0, 1, samples_per_entity):
                        pt = curve.point(t)
                        points.append([pt.x, pt.y, pt.z])
                except Exception as e:
                    print(f"Could not sample {entity.dxftype()}: {e}")
                    continue

        except Exception as err:
            print(f"Skipping entity {entity.dxftype()} due to error: {err}")
            continue

    return np.array(points)


def generate_kerf_dxf(control_points, curve_type, tool_type, cone_angle, cut_width, cut_depth, line_length, offset, output_file, arc_radius, arc_angle, dxf_path,
                      display_extra_geometries=False, search_window=80, curve_samples=None, spline_tension=0.9, desired_sample_length=1.5, algo="default"):
    """Generate a kerf pattern DXF file and return cut information"""
    # Calculate "kerf angle" in radians - the angle by which a single cut bends the material

    if tool_type == "saw": kerf_angle_rad = 2 * atan2(cut_width, 2 * cut_depth)
    else:  
        kerf_angle_rad = radians(cone_angle)  
        cut_width = 2*tan(kerf_angle_rad/2)*cut_depth

    # Initialize DXF document
    doc = ezdxf.new(dxfversion='AC1027', setup=False, units=6)  #these are the default, same as just ezdxf.new(), AC1027 is AutoCAD R2013
    msp = doc.modelspace()
    layers = doc.layers
    layers.add(name="CutsSideA", color=5)
    layers.add(name="CutsSideB", color=1)
    layers.add(name="Start and end of workpiece", color=6)
    if display_extra_geometries:
        layers.add(name="DebugA", color=3)  # Indicators for side A cuts
        layers.add(name="DebugB", color=5)  # Indicators for side B cuts
        layers.add(name="Original Curve", color=5)
    # Process control points
    # Convert list of [x,y] to list of (x,y) if needed
    cpoints = []
    for p in control_points:
        if isinstance(p, list):
            cpoints.append((p[0], p[1]))
        else:
            cpoints.append(p)

    # Generate curve points
    cuts_distances=[]
    side_list=[]
    message = []
    #---------------------------------- ARC HANDLING--------------------------------------------
    if curve_type == "arc":
        # For arc curve type
        total_length = radians(arc_angle) * arc_radius
        num_cuts = int(ceil(radians(arc_angle)/kerf_angle_rad))
        
        cuts_distances.append(0)
        side_list.append('A')
        for i in range(num_cuts-1):
            cuts_distances.append((radians(arc_angle)*arc_radius)/(num_cuts-1))
            side_list.append('A')
        side_list.append('A')
        
        display_extra_geometries=False
    
        #current_x = -(sum(cuts_distances)/2)
        current_x=0

        # Add the starting edge
        msp.add_line((current_x, -15), (current_x, line_length+15), dxfattribs={"layer": "Start and end of workpiece"})

        # Add the cuts
        for i in range(len(cuts_distances)):
            current_x += cuts_distances[i]
            msp.add_line((current_x+offset, 0), (current_x+offset, line_length), dxfattribs={"layer": "CutsSideA"})

        # Add the final right edge
        msp.add_line((current_x, -15), (current_x, line_length+15), dxfattribs={"layer": "Start and end of workpiece"})

        # Check for cuts that are too close
        if (radians(arc_angle)*arc_radius)/(num_cuts-1) < 2 * cut_width:
            message = (f"WARNING: Some cuts are closer than recommended, minimum is {cuts_distances[i]:.2f}, recommended is > {2*cut_width:.2f} (twice the cut width). Consider reducing the radius of curvature to resolve this issue.")

        msp.add_text(
            f"Width: {cut_width}, Depth: {cut_depth}",
            height=20,
            dxfattribs={"style": "LiberationSerif"}
        ).set_placement((0, -150), align=TextEntityAlignment.CENTER)
        
        # Save the file
        doc.saveas(output_file)
        # cuts_distances[:-1]
        return cuts_distances, total_length, num_cuts, side_list, message, "Arc"
    # --------------------------------------------CURVE HANDLING INITIALIZATION--------------------------------------------------------

    if algo == "default":           #we need to check for mixed curvature curves by seeing if all the c_point are above or below the 
        if has_double_curvature(control_points): algo = "length_diff" 
        else: algo="ray_reflection"

    #print(algo)
    
    # For bezier or spline curve types, estimate length first for adaptive sampling

    if curve_type != 'dxf' :
        estimated_length = estimate_curve_length(cpoints, curve_type, spline_tension)
        #print(f"Estimated curve length: {estimated_length:.2f} units")
        
        if algo == "ray_reflection":
            curve_samples = max(500, int(estimated_length / (desired_sample_length*2)))   #uses half as many samples
            curve_samples = 2*int(curve_samples/2)  # make it even
            if curve_samples > 1500: curve_samples = 1500
        else:
            curve_samples = max(1000, int(estimated_length / desired_sample_length))
            curve_samples = 2*int(curve_samples/2)  # make it even
            if curve_samples > 2000: curve_samples = 2000
            #print(f"Using {curve_samples} samples for { desired_sample_length} unit per sample")
        # Generate curve points
        curve_points = create_curve_points(cpoints, curve_samples, curve_type, spline_tension)

    else: 
        inputdoc = ezdxf.readfile(dxf_path)
        inputmsp = inputdoc.modelspace()
        curve_points = import_dxf(inputmsp, 7)
        #if len(curve_points)<700: import_dxf(inputmsp, 14)

    if display_extra_geometries:
        # Draw control points
        #for point in cpoints:
        #    # Add z=0 if needed
        #    point_3d = (point[0], point[1], 0) if len(point) == 2 else point
        #    msp.add_point(point_3d, dxfattribs={"color": 3})
        # Draw curve
        draw_curve(msp, curve_points, 5)

    #---------------------------------------------------CORE ALGO-------------------------------------

    #since different algos suit different curve types better I will give the user the option to choose


            
#_____________________________________________________LENGHT DIFFERENCE ALGO USED FOR COMPLEX GEOMETRIES________________________________________________
    if algo == "length_diff":
        offset_curve_points = offset_curve(curve_points, cut_depth)
        if display_extra_geometries:
            draw_curve(msp, offset_curve_points, 1)

  
        # Step 2: Walk and compare length deltas
        original_len = 0
        offset_len = 0
        side_list = []
        cut_points =[]

        cuts = [0]
        side_list = ['Undefined']

        for i in range(1, len(curve_points)):
            p1_orig = np.array(curve_points[i-1][:2])
            p2_orig = np.array(curve_points[i][:2])
            p1_off = np.array(offset_curve_points[i-1][:2])
            p2_off = np.array(offset_curve_points[i][:2])

            original_segment = np.linalg.norm(p2_orig - p1_orig)    #numpy norm computes distance between points
            offset_segment = np.linalg.norm(p2_off - p1_off)

            original_len += original_segment        #distance is added to total
            offset_len += offset_segment

            if abs(original_len - offset_len) >= cut_width:
                cuts.append(i)
                side_list.append("A" if offset_len < original_len else "B")
                original_len = 0
                offset_len = 0

        cuts.append(len(curve_points)-1)
        side_list.append('Undefined')

        for j in cuts:
            cut_points.append(curve_points[j])
        #------------------------------------- -----BOUNDARY CUT REMOVAL---------------------------------------

        #print(f"Before removing boundary cuts: {len(cut_points)} cut points")
        #cut_points, side_list = remove_boundary_cuts(cut_points, side_list)  #filter the arrays to remove buondary cuts
        #print(f"After removing boundary cuts: {len(cut_points)} cut points")
                    

        #------------------------------------------ DEBUG GEOMETRIES-------------------------------
        # If showing extra geometries
        if display_extra_geometries:
            # Draw curve through all cut points
            draw_curve(msp, cut_points, 1)
        
            for i, idx in enumerate(cut_points):
                layer = "DebugA" if side_list[i] == 'A' else "DebugB"
                msp.add_point(idx, dxfattribs={"layer": layer})

        #------------------------------------------------  LAST CHECKS AND FINAL RENDERING----------------------------------

        # Calculate distances between adjacent cuts
        cuts_distances = [two_p_distance(cut_points[i], cut_points[i+1]) for i in range(len(cut_points) - 1)]
                        
        for i in range(len(cuts_distances)):    
            if side_list[i] == "B":              
                cuts_distances[i] += cut_width      #basically due to the nature of what we are doing (cutting using a saw) the side that gets cut with the saw gets 
                                                    #shortened, I add cut_width to compensate 😖 hard to understand, quite counterintuitive
                                                    #THIS IS ACTUALLY KEY TO THE OVERALL ALGO, without this the results for cutSideB are incorrect

        # Check for cuts that are too close
        for i in range(len(cuts_distances)-1):
            if cuts_distances[i] < 2 * cut_width:
                message = (f"WARNING: Some cuts are closer than recommended, minimum is {cuts_distances[i]:.2f}, recommended is > {2*cut_width:.2f} (twice the cut width). Consider reducing the radius of curvature to resolve this issue.")
                break

        # Draw the final cut lines
        #current_x = -(sum(cuts_distances)/2)
        current_x = 0

        # Add the starting edge
        msp.add_line((current_x, 0), (current_x, line_length+20), dxfattribs={"layer": "Start and end of workpiece"})

        # Add the cuts
        #print(len(cuts_distances))
        #print(len(side_list))
        for i in range(len(cuts_distances)):    
            current_x += cuts_distances[i] 

            # Determine the layer based on side
            if i+1 < len(cut_points):
                if side_list[i+1] == 'B':
                    layer = "CutsSideB"
                elif side_list[i+1] == 'A':
                    layer = "CutsSideA"
                else:
                    #layer = "DebugA"         #this is for debug purpouses only and should never trigger
                    break
                # Draw the cut line
                msp.add_line((current_x+offset, 0), (current_x+offset, line_length), dxfattribs={"layer": layer})

        # Add the final right edge
        #msp.add_line((sum(cuts_distances)/2, 0), (sum(cuts_distances)/2, line_length+20), dxfattribs={"layer": "Start and end of workpiece"})
        msp.add_line((sum(cuts_distances), 0), (sum(cuts_distances), line_length+20), dxfattribs={"layer": "Start and end of workpiece"})

        msp.add_text(
            f"Width: {cut_width}, Depth: {cut_depth}",
            height=20,
            dxfattribs={"style": "LiberationSerif"}
        ).set_placement((0, -150), align=TextEntityAlignment.CENTER)
        # Save the file
        doc.saveas(output_file)
        total_length = sum(cuts_distances)
        num_cuts = len(cut_points)-2
        
        #print(f"kerf angle: {kerf_angle_rad}")

        return cuts_distances[:-1], total_length, num_cuts, side_list, message, "Primary"

#_____________________________________________________THE RAY REFLECTION ALGO USED FOR SIMPLE GEOMETRIES________________________________________________
    
    #if algo == "ray_reflection":
    else:
        cut_points = []
        side_list = []

        backward_cut_points = []
        backward_sides = []

        forward_cut_points = []
        forward_sides = []

        # Find middle index of the curve
        middle_idx = len(curve_points) // 2

        # Calculate tangent using points on both sides for stability
        point_before = curve_points[middle_idx - 1]
        point_after = curve_points[middle_idx + 1]
        middle_point = curve_points[middle_idx]

        # Calculate tangent vector using both neighboring points
        vec_before = np.array(middle_point) - np.array(point_before)
        vec_after = np.array(point_after) - np.array(middle_point)
        tangent_vector = vec_before + vec_after  # Average direction

        # Normalize the tangent vector
        tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)

        # Calculate the tangent angle
        middle_tangent_angle = np.arctan2(tangent_vector[1], tangent_vector[0])
        #----------------------------------NEW ALGO------------------------------------------------------
        A = curve_points[middle_idx]      # Middle point as reference
        #print(f"tangent should be 0 is {middle_tangent_angle}")

        ray_angle = middle_tangent_angle
        initial_offset = kerf_angle_rad/2
        use_initial_offset = True
        
        for id in range(middle_idx+1, len(curve_points)-1):
            # Calculate angles
            angle_up = ray_angle + (initial_offset if use_initial_offset else kerf_angle_rad)
            angle_down = ray_angle - (initial_offset if use_initial_offset else kerf_angle_rad)
            # Find intersections
            intersectionUP = ray_segment_intersection(A, angle_up, curve_points[id], curve_points[id+1])
            intersectionDOWN = ray_segment_intersection(A, angle_down, curve_points[id], curve_points[id+1])

            if intersectionDOWN and intersectionUP:     #this handles enge cases where both UP and DOWN are found, for non monotone curves
                if find_nearest_index(intersectionUP, curve_points) > find_nearest_index(intersectionDOWN, curve_points):
                    intersectionUP = None
                else: intersectionDOWN = None

            if intersectionUP:
                if display_extra_geometries:  draw_line(msp, A[0], A[1], angle_up, 300)
                forward_cut_points.append(intersectionUP)
                forward_sides.append('A')
                ray_angle += initial_offset if use_initial_offset else kerf_angle_rad
                A = intersectionUP
                use_initial_offset = False  # Switch to full angle after first intersection
            elif intersectionDOWN:
                if display_extra_geometries: draw_line(msp, A[0], A[1], angle_down, 300)
                forward_cut_points.append(intersectionDOWN)
                forward_sides.append('B')
                ray_angle -= initial_offset if use_initial_offset else kerf_angle_rad
                A = intersectionDOWN
                use_initial_offset = False

        

                #GO BACKWARDS
        A = curve_points[middle_idx]
        ray_angle = math.pi + middle_tangent_angle
        initial_offset = kerf_angle_rad/2
        use_initial_offset = True

        for id in reversed(range(1 , middle_idx-1)):
            # Calculate angles
            angle_up = ray_angle + (initial_offset if use_initial_offset else kerf_angle_rad)
            angle_down = ray_angle - (initial_offset if use_initial_offset else kerf_angle_rad)
        
            # Find intersections
            intersectionUP = ray_segment_intersection(A, angle_up, curve_points[id], curve_points[id-1])
            intersectionDOWN = ray_segment_intersection(A, angle_down, curve_points[id], curve_points[id-1])

            if intersectionDOWN and intersectionUP:     #this handles enge cases where both UP and DOWN are found, for non monotone curves
                if find_nearest_index(intersectionUP, curve_points) > find_nearest_index(intersectionDOWN, curve_points):
                    intersectionUP = None
                else: intersectionDOWN = None

            if intersectionUP:
                if display_extra_geometries:  draw_line(msp, A[0], A[1], angle_up, 300)
                backward_cut_points.append(intersectionUP)
                backward_sides.append('B')
                ray_angle += initial_offset if use_initial_offset else kerf_angle_rad
                A = intersectionUP
                use_initial_offset = False  # Switch to full angle after first intersection
            elif intersectionDOWN:
                if display_extra_geometries:  draw_line(msp, A[0], A[1], angle_down, 300)
                backward_cut_points.append(intersectionDOWN)
                backward_sides.append('A')
                ray_angle -= initial_offset if use_initial_offset else kerf_angle_rad
                A = intersectionDOWN
                use_initial_offset = False
        
        #-----------------stil useful after rewrite if we still want ot "enforce simmetry"--------------------------------------------

        # Determine side for middle point based on curvature
        if middle_idx > 1 and middle_idx < len(curve_points) - 1:
            # Calculate vectors before and  after middle point
            vec_before = np.array(curve_points[middle_idx]) - np.array(curve_points[middle_idx - 1])
            vec_after = np.array(curve_points[middle_idx + 1]) - np.array(curve_points[middle_idx])
            
            # Calculate angle between the vectors
            angle = signed_angle_between(vec_before, vec_after)
            
            # If angle is positive (turning left/counterclockwise), assign side A
            # If angle is negative (turning right/clockwise), assign side B
            middle_side = 'A' if angle >= 0 else 'B'
        else:
            # Fallback in case middle_idx is at the boundary
            middle_side = 'A'

        # -----------------------------------we want to get here with 2 arrays: forward_cut_points, backward_cut_points
    
        # Combine results in the correct order: start -> backward cuts -> middle -> forward cuts -> end
        # Start with the first point
        cut_points = [curve_points[0]]
        side_list = ['Undefined']
        cuts = []

        # Add backward cuts (in reverse order, to maintain start -> middle direction)
        #for i in range(len(backward_cut_points) - 1, -1, -1):
        for cut_point, side in zip(reversed(backward_cut_points), reversed(backward_sides)):
            cut_points.append(cut_point)
            side_list.append(side)
            #cuts.append(backward_cuts[i])

        # Add middle point
        cut_points.append(curve_points[middle_idx])
        side_list.append(middle_side)
        #cuts.append(middle_idx)

        # Add forward cuts
        for i in range(len(forward_cut_points)):
            cut_points.append(forward_cut_points[i])
            side_list.append(forward_sides[i])
            #cuts.append(forward_cuts[i])

        # Add the last point
        cut_points.append(curve_points[-1])
        side_list.append('Undefined')
        #------------------------------------------BOUNDARY CUT REMOVAL---------------------------------------

        #print(f"Before removing boundary cuts: {len(cut_points)} cut points")
        #cut_points, side_list = remove_boundary_cuts(cut_points, side_list)  #filter the arrays to remove buondary cuts
        #print(f"After removing boundary cuts: {len(cut_points)} cut points")
                    

        #------------------------------------------OVERLY COMPLEX DEBUG GEOMETRIES-------------------------------
        # If showing extra geometries
        if display_extra_geometries:
            # Draw curve through all cut points
            draw_curve(msp, cut_points, 1)
            
            # Draw lines at cut points with different colors for side A and B
            # All of this could also just be done in a single line,  msp.add_point
            for index, point in enumerate(cut_points):
                try: layer = "DebugA" if side_list[index+1] == 'A' else "DebugB"
                except: layer = "DebugA" if side_list[index] == 'A' else "DebugB"
                msp.add_point(point, dxfattribs={"layer": layer})
                

        #------------------------------------------------  LAST CHECKS AND FINAL RENDERING----------------------------------
        # Calculate distances between adjacent cuts
        cuts_distances = [two_p_distance(cut_points[i], cut_points[i+1]) for i in range(len(cut_points) - 1)]
                        
        for i in range(len(cuts_distances)):    
            if side_list[i] == "B":              
                cuts_distances[i] += cut_width      #basically due to the nature of what we are doing (cutting using a saw) the side that gets cut with the saw gets 
                                                    #shortened, I add cut_width to compensate 😖 hard to understand, quite counterintuitive
                                                    #THIS IS ACTUALLY KEY TO THE OVERALL ALGO, without this the results for cutSideB are incorrect

        # Check for cuts that are too close
        for i in range(len(cuts_distances)-1):
            if cuts_distances[i] < 2 * cut_width:
                message = (f"WARNING: Some cuts are closer than recommended, minimum is {cuts_distances[i]:.2f}, recommended is > {2*cut_width:.2f} (twice the cut width). Consider reducing the radius of curvature to resolve this issue.")
                break

        # Draw the final cut lines
        #current_x = -(sum(cuts_distances)/2)
        current_x = 0

        # Add the starting edge
        msp.add_line((current_x, 0), (current_x, line_length+20), dxfattribs={"layer": "Start and end of workpiece"})

        # Add the cuts
        #print(len(cuts_distances))
        #print(len(side_list))
        for i in range(len(cuts_distances)):    
            current_x += cuts_distances[i] 

            # Determine the layer based on side
            if i+1 < len(cut_points):
                if side_list[i+1] == 'B':
                    layer = "CutsSideB"
                elif side_list[i+1] == 'A':
                    layer = "CutsSideA"
                else:
                    #layer = "DebugA"         #this is for debug purpouses only and should never trigger
                    break
                # Draw the cut line
                msp.add_line((current_x+offset, 0), (current_x+offset, line_length), dxfattribs={"layer": layer})

        # Add the final right edge
        msp.add_line((current_x, 0), (current_x, line_length+20), dxfattribs={"layer": "Start and end of workpiece"})


        msp.add_text(
            f"Width: {cut_width}, Depth: {cut_depth}",
            height=20,
            dxfattribs={"style": "LiberationSerif"}
        ).set_placement((0, -150), align=TextEntityAlignment.CENTER)


        # Save the file
        doc.saveas(output_file)
        total_length = sum(cuts_distances)
        num_cuts = len(cut_points)-2
        
        #print(f"kerf angle: {kerf_angle_rad}")

        return cuts_distances[:-1], total_length, num_cuts, side_list, message, "Simple-symmetric"


if debug==True:
    debug_distances, debug_length, debug_numcuts, debug_sidelist, mess, algo=(generate_kerf_dxf([(-400,0),(0, -600),(400, 0),(800, -550)],
                                  "spline", "saw",
                                  5, 2.7,
                                  7, 80,
                                  0, "output_file_DEBUG.dxf",
                                  arc_radius = 300,
                                  arc_angle=90,
                                  display_extra_geometries=True,
                                  search_window=80,         #now unused
                                  curve_samples=None,  # now determined adaptively
                                  spline_tension=0.9,
                                  dxf_path = "placeholder",
                                  desired_sample_length=50,   #this is the new way to handle changing the sampling accuracy, but it's still clamped to a min value
                                  algo="length_diff")     #enum: lenght_diff, ray_reflection, default
                      )
    print(f"distances: {debug_distances}")
    print(f"length: {debug_length:.0f}mm")
    print(f"numcuts: {debug_numcuts}")
    print(f"sides: {debug_sidelist}")
    
    if mess:
        print(f"Message: {mess}")
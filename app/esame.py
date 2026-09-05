from flask import Flask, send_from_directory, request, jsonify, send_file, render_template, redirect, url_for, Response
from flask_babel import Babel
import tempfile
import base64
import os
from flask_cors import CORS
from werkzeug.utils import safe_join
import uuid
from apscheduler.schedulers.background import BackgroundScheduler

debug = False

def get_locale():
    if 'lang' in request.args:  # Check URL param first
        return request.args['lang'] if request.args['lang'] in ['en', 'it','es'] else 'en'
    return request.accept_languages.best_match(['en', 'it','es']) or 'en'

app = Flask(__name__,
            static_folder='../static',
            template_folder='../templates')

@app.context_processor
def inject_lang():
    return {'current_lang': request.args.get('lang', 'en')}

CORS(app)  # Enable CORS for all routes


babel = Babel(app, locale_selector=get_locale)
app.jinja_env.add_extension('jinja2.ext.i18n')

from .kerf_backend import generate_kerf_dxf

# Directory to store temporary files
TEMP_DIR = tempfile.gettempdir()
os.makedirs(TEMP_DIR, exist_ok=True)
#print(TEMP_DIR)

@app.before_request
def ensure_lang_param():
    if 'lang' not in request.args:
        lang = get_locale()
        # Rebuild URL with lang parameter
        new_url = url_for(
            request.endpoint,
            lang=lang,
            **dict(request.args.to_dict(flat=False), **request.view_args)
        )

        return redirect(new_url, code=307)  # 307 preserves POST data


@app.route('/robots.txt')
def robots_txt():
    robots_content = "User-agent: *\nDisallow:"
    return Response(robots_content, mimetype='text/plain')

@app.route('/ads.txt')
def ads_txt():
    ads_content = "User-agent: *\nDisallow:"
    return Response(ads_content, mimetype='text/plain')


@app.route('/')
def home():
    # Get the absolute path to the static directory
    return render_template('temporaneo.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)


"""@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'), 
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon')   """


@app.route('/api/generate-kerf', methods=['POST'])
def generate_kerf():
    try:
        data = request.json
        # Extract parameters from request
        control_points = data['controlPoints']
        curve_type = data['curveType']  # "bezier" or "spline"
        tool_type = data['toolType']   # saw or cone
        cut_width = float(data['cutWidth'])
        cone_angle = float(data['coneAngle'])
        cut_depth = float(data['cutDepth'])
        line_length = float(data['lineLength'])
        offset = float(data['offset'])
        arc_radius = float(data['arcRadius'])
        arc_angle = float(data['arcAngle'])
        display_extra_geometries = data['extraGeometries']
        algo_type = data['algo']
        dxf64 = data['dxfBase64']

        # Optional parameters with defaults
        search_window = int(data.get('searchWindow', 40))
        curve_samples = int(data.get('curveSamples', 600))
        spline_tension = float(data.get('splineTension', 0.9))

        if curve_type == 'dxf':
            # Decode the base64 DXF
            b64data = dxf64
            dxf_bytes = base64.b64decode(b64data)
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
                tmp.write(dxf_bytes)
                tmp_path = tmp.name
        else: tmp_path = "empty"

        # Generate a unique filename for output
        unique_id = str(uuid.uuid4())[:8]
        temp_dxf_path = os.path.join(TEMP_DIR, f"kerf_{unique_id}.dxf")

        if debug == True: print(data)

        # Call the kerf function
        cut_distances, total_length, num_cuts, side_list, message, algo_msg = generate_kerf_dxf(
            control_points=control_points,
            curve_type=curve_type,
            tool_type=tool_type,
            cone_angle=cone_angle,
            cut_width=cut_width,
            cut_depth=cut_depth,
            line_length=line_length,
            offset=offset,
            output_file=temp_dxf_path,
            display_extra_geometries=display_extra_geometries,
            search_window=search_window,
            curve_samples=curve_samples,
            spline_tension=spline_tension,
            arc_radius=arc_radius,
            arc_angle=arc_angle,
            algo = algo_type,
            dxf_path = tmp_path
        )
        # Return both the file and the cut information
        result = {
            'cutDistances': cut_distances,
            'totalLength': total_length,
            'numCuts': num_cuts,
            'sideList': side_list,
            'message': message,
            'algo_msg': algo_msg
            #'kerfAngle': 2 * atan2(cut_width, 2 * cut_depth) * 180/pi  # in degrees
        }

        # Return the filename for download endpoint
        return jsonify({
            'result': result,
            'dxfPath': os.path.basename(temp_dxf_path)
        })

    except Exception as e:
        print(f"error in API: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/download-dxf/<filename>', methods=['GET'])
def download_dxf(filename):
    try:
        # Ensure the filename is safe
        filepath = safe_join(TEMP_DIR, filename)

        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        # Serve the file
        return send_file(
            filepath,
            as_attachment=True,
            #attachment_filename=filename,
            mimetype='application/dxf'
        )

    except Exception as e:
        return jsonify({
            'API error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)


import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file, session
import smtplib
import ssl
from flask_mysqldb import MySQL
from flask_socketio import SocketIO
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from MySQLdb.cursors import DictCursor
from rfp_analyzer_routes import rfp_bp 
# --- App Initialization ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file upload

# Register RFP Analyzer Blueprint
app.register_blueprint(rfp_bp)

# --- Stage Constants ---
PIPELINE = ['analyzer', 'business', 'design', 'operations', 'engineer', 'handover']
LABELS = {
    'analyzer': 'BID Analyzer',
    'business': 'Business Development', 
    'design': 'Design Team',
    'operations': 'Operations Team',
    'engineer': 'Site Engineer',
    'handover': 'Handov'
}

def pct_for(stage: str) -> int:
    """Calculate progress percentage based on stage"""
    s = (stage or 'analyzer').lower()
    i = PIPELINE.index(s) if s in PIPELINE else 0
    return int(round(i * (100 / (len(PIPELINE) - 1))))  # 0,20,40,60,80,100

def status_texts(stage: str) -> tuple[str, str]:
    """Generate project status and work status texts based on stage"""
    s = (stage or 'analyzer').lower()
    proj = 'completed' if s == 'handover' else 'ongoing'
    if s == 'analyzer':
        work = 'Initiated by BID Analyzer'
    else:
        i = PIPELINE.index(s) if s in PIPELINE else 0
        prev = PIPELINE[i-1] if i > 0 else None
        from_txt = LABELS.get(prev, '').replace(' Team', '')
        to_txt = LABELS.get(s, '')
        work = f'Updated by {from_txt} to {to_txt}'
    return proj, work

def generate_team_checklist(cur, g_id, team):
    """Generate team-specific default checklist tasks"""
    team_tasks = {
        'business': [
            {'name': 'Market Analysis', 'description': 'Analyze market potential and competition', 'priority': 'high'},
            {'name': 'Client Communication', 'description': 'Establish communication with client', 'priority': 'high'},
            {'name': 'Proposal Preparation', 'description': 'Prepare business proposal', 'priority': 'medium'},
            {'name': 'Budget Estimation', 'description': 'Estimate project budget and timeline', 'priority': 'high'}
        ],
        'design': [
            {'name': 'Initial Design Concept', 'description': 'Create initial design concepts', 'priority': 'high'},
            {'name': 'Technical Drawings', 'description': 'Prepare detailed technical drawings', 'priority': 'high'},
            {'name': 'Material Selection', 'description': 'Select appropriate materials and specifications', 'priority': 'medium'},
            {'name': 'Design Review', 'description': 'Review and finalize design with team', 'priority': 'medium'},
            {'name': 'Client Approval', 'description': 'Get client approval on design', 'priority': 'high'}
        ],
        'operations': [
            {'name': 'Project Planning', 'description': 'Create detailed project execution plan', 'priority': 'high'},
            {'name': 'Resource Allocation', 'description': 'Allocate resources and personnel', 'priority': 'high'},
            {'name': 'Timeline Management', 'description': 'Set up project timeline and milestones', 'priority': 'medium'},
            {'name': 'Quality Control Setup', 'description': 'Establish quality control procedures', 'priority': 'medium'},
            {'name': 'Risk Assessment', 'description': 'Identify and assess project risks', 'priority': 'high'}
        ],
        'engineer': [
            {'name': 'Site Survey', 'description': 'Conduct detailed site survey', 'priority': 'high'},
            {'name': 'Technical Specifications', 'description': 'Prepare technical specifications', 'priority': 'high'},
            {'name': 'Safety Planning', 'description': 'Develop safety protocols and procedures', 'priority': 'high'},
            {'name': 'Equipment Planning', 'description': 'Plan equipment and tool requirements', 'priority': 'medium'},
            {'name': 'Implementation Plan', 'description': 'Create detailed implementation plan', 'priority': 'high'}
        ]
    }
    
    if team in team_tasks:
        for task in team_tasks[team]:
            # Persist a stage on each seeded task so aggregation can be stage-aware
            stage_name = team.strip().lower()
            cur.execute("""
                INSERT INTO bid_checklists (g_id, task_name, description, priority, status, progress_pct, stage, created_by)
                VALUES (%s, %s, %s, %s, 'pending', %s, %s, %s)
            """, (g_id, task['name'], task['description'], task['priority'], 0, stage_name, current_user.id))

def log_write(action: str, details: str = ''):
    """Write to logs table and emit via Socket.IO"""
    try:
        cur = mysql.connection.cursor()
        user_id = getattr(current_user, 'id', None) if hasattr(current_user, 'id') else None
        cur.execute("INSERT INTO logs (action, user_id) VALUES (%s, %s)",
                    (f"{action} | {details}", user_id))
        mysql.connection.commit()
        cur.close()
        
        # Emit to master dashboard
        socketio.emit('master_update', {'log': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'action': f"{action} | {details}",
            'user_email': getattr(current_user, 'email', 'System') if hasattr(current_user, 'email') else 'System',
            'user_role': getattr(current_user, 'role', '') if hasattr(current_user, 'role') else ''
        }})
    except Exception as e:
        print(f"Log write error: {e}")

# Legacy function for backward compatibility
def stage_progress_pct(stage: str) -> int:
    return pct_for(stage)

# MySQL Config
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'esco'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

socketio = SocketIO(app)

@app.route('/profiling', methods=['GET', 'POST'])
@login_required
def profiling():
    if not current_user.is_admin:
        return "Access Denied", 403
    # Knowledge Hub data (embedded tab)
    base_dir = os.path.join(os.getcwd(), 'uploads', 'knowledge')
    os.makedirs(base_dir, exist_ok=True)
    message = ''
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'create_folder':
            folder_name = (request.form.get('folder_name') or '').strip()
            if folder_name:
                target = os.path.join(base_dir, folder_name)
                os.makedirs(target, exist_ok=True)
                message = f'Folder "{folder_name}" created.'
        elif action == 'upload_docs':
            target_folder = (request.form.get('target_folder') or '').strip()
            target_path = os.path.join(base_dir, target_folder) if target_folder else base_dir
            os.makedirs(target_path, exist_ok=True)
            files = request.files.getlist('documents')
            for f in files:
                if not f.filename:
                    continue
                safe_name = f.filename.replace('..','_')
                f.save(os.path.join(target_path, safe_name))
            message = 'Documents uploaded.'

    folders = []
    files = []
    for entry in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, entry)
        if os.path.isdir(p):
            folders.append(entry)
        else:
            files.append(entry)

    return render_template('profiling.html', folders=folders, files=files, message=message)

# --- Ensure seed tasks exist for every team (parallel kickoff) ---
def ensure_tasks_for_team(team: str):
    cur = mysql.connection.cursor(DictCursor)
    try:
        cur.execute(
            """
            SELECT gb.g_id
            FROM go_bids gb
            LEFT JOIN bid_checklists bc
              ON bc.g_id = gb.g_id AND LOWER(COALESCE(bc.stage,'')) = %s
            WHERE bc.id IS NULL
            """, (team,)
        )
        missing = [row['g_id'] for row in cur.fetchall()]
        if missing:
            for g_id in missing:
                generate_team_checklist(cur, g_id, team)
            mysql.connection.commit()
    except Exception:
        mysql.connection.rollback()
    finally:
        cur.close()

@app.route('/knowledge', methods=['GET', 'POST'])
@login_required
def knowledge():
    if not current_user.is_admin:
        return "Access Denied", 403
    base_dir = os.path.join(os.getcwd(), 'uploads', 'knowledge')
    os.makedirs(base_dir, exist_ok=True)

    message = ''
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'create_folder':
            folder_name = (request.form.get('folder_name') or '').strip()
            if folder_name:
                target = os.path.join(base_dir, folder_name)
                os.makedirs(target, exist_ok=True)
                message = f'Folder "{folder_name}" created.'
        elif action == 'upload_docs':
            target_folder = (request.form.get('target_folder') or '').strip()
            target_path = os.path.join(base_dir, target_folder) if target_folder else base_dir
            os.makedirs(target_path, exist_ok=True)
            files = request.files.getlist('documents')
            for f in files:
                if not f.filename:
                    continue
                safe_name = f.filename.replace('..','_')
                f.save(os.path.join(target_path, safe_name))
            message = 'Documents uploaded.'

    # List folders and files
    folders = []
    files = []
    for entry in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, entry)
        if os.path.isdir(p):
            folders.append(entry)
        else:
            files.append(entry)

    return render_template('profiling.html', folders=folders, files=files, message=message)
# --- Helper: Assign bids by revenue into assigned_bids ---
def assign_bids_by_revenue():
    cur = mysql.connection.cursor(DictCursor)
    try:
        # Ensure table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS assigned_bids (
                id INT AUTO_INCREMENT PRIMARY KEY,
                g_id INT,
                b_name VARCHAR(100),
                type VARCHAR(100),
                revenue DECIMAL(15,2) DEFAULT 0.00,
                assigned_to VARCHAR(100),
                assigned_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Align schema: drop company column if exists; add type if missing
        try:
            cur.execute("""
                SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME='assigned_bids' AND COLUMN_NAME='company'
            """)
            if cur.fetchone():
                cur.execute("ALTER TABLE assigned_bids DROP COLUMN company")
        except Exception:
            pass
        try:
            cur.execute("""
                SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME='assigned_bids' AND COLUMN_NAME='type'
            """)
            if not cur.fetchone():
                cur.execute("ALTER TABLE assigned_bids ADD COLUMN type VARCHAR(100)")
        except Exception:
            pass

        # Pull all go_bids with revenue and type
        cur.execute("""
            SELECT g_id, b_name, COALESCE(type,'') AS type, COALESCE(revenue,0) AS revenue
            FROM go_bids
            WHERE COALESCE(revenue,0) > 0
            ORDER BY revenue DESC
        """)
        rows = cur.fetchall()
        if not rows:
            cur.close()
            return

        def upsert(g_id, b_name, type_val, revenue, assignee):
            cur2 = mysql.connection.cursor(DictCursor)
            cur2.execute("SELECT id FROM assigned_bids WHERE g_id=%s", (g_id,))
            row = cur2.fetchone()
            if row:
                cur2.execute(
                    "UPDATE assigned_bids SET b_name=%s, type=%s, revenue=%s, assigned_to=%s, assigned_at=NOW() WHERE id=%s",
                    (b_name, type_val, revenue, assignee, row['id'])
                )
            else:
                cur2.execute(
                    "INSERT INTO assigned_bids (g_id, b_name, type, revenue, assigned_to, assigned_at) VALUES (%s,%s,%s,%s,%s,NOW())",
                    (g_id, b_name, type_val, revenue, assignee)
                )
            mysql.connection.commit()
            cur2.close()

        # Assign based on revenue ranges
        for r in rows:
            rev = float(r['revenue'] or 0)
            type_val = (r['type'] or '').lower()
            # Type-based routing first
            if 'solar' in type_val:
                assignee = 'Rahul'
            else:
                # Default lighting (or other) rules by revenue
                if rev >= 1000:
                    assignee = 'Chris'
                elif rev >= 500:
                    assignee = 'Inder'
                elif rev >= 1:
                    assignee = 'Gurki'
                else:
                    continue
            upsert(r['g_id'], r['b_name'], r['type'], r['revenue'], assignee)
    finally:
        try:
            cur.close()
        except Exception:
            pass

# assign_go_bids deprecated
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Custom User class
class User(UserMixin):
    def __init__(self, id, email, password, is_admin, role):
        self.id = id
        self.email = email
        self.password = password
        self.is_admin = bool(is_admin)
        self.role = role
    
    @property
    def is_supervisor(self):
        return self.role.lower() == 'supervisor'
    
    @property
    def can_assign_stages(self):
        return self.is_admin or self.is_supervisor
    
    @property
    def can_alter_timeline(self):
        return self.is_admin or self.is_supervisor

# --- User Loader for Flask-Login ---
@login_manager.user_loader
def load_user(user_id):
    cur = mysql.connection.cursor(DictCursor)
    cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
    data = cur.fetchone()
    cur.close()
    if data:
        return User(data['id'], data['email'], data['password'], data['is_admin'], data['role'])
    return None

# --- Authentication Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        cur = mysql.connection.cursor(DictCursor)
        cur.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
        user = cur.fetchone()
        if user:
            cur.close()
            user_obj = User(user['id'], user['email'], user['password'], user['is_admin'], user['role'])
            login_user(user_obj)
            log_write('login', f"role={user['role']}")
            if user['is_admin']:
                return redirect(url_for('master_dashboard'))
            
            # Role-based redirects
            role = user['role'].lower()
            if role == 'supervisor':
                return redirect(url_for('supervisor_dashboard'))
            elif role == 'business dev':
                return redirect(url_for('team_dashboard', team='business'))
            elif role == 'design':
                return redirect(url_for('team_dashboard', team='design'))
            elif role == 'operations':
                return redirect(url_for('team_dashboard', team='operations'))
            elif role == 'site manager':
                return redirect(url_for('team_dashboard', team='engineer'))
            else:
                return redirect(url_for('dashboard'))
        # Fallback: try authenticating as employee
        try:
            # Ensure password column exists before querying
            cur.execute("SHOW COLUMNS FROM employees LIKE 'password'")
            if cur.fetchone() is None:
                cur.close()
                return 'Employee login unavailable: employees.password missing', 500
            cur.execute(
                """
                SELECT * FROM employees
                WHERE email=%s AND password=%s AND is_active=1
                """,
                (email, password)
            )
            employee = cur.fetchone()
        finally:
            cur.close()

        if employee:
            session['employee_id'] = employee['id']
            session['employee_name'] = employee['name']
            session['employee_email'] = employee['email']
            session['employee_department'] = employee['department']
            log_write('employee_login', f"Employee {employee['name']} logged in via /login")
            return redirect(url_for('employee_dashboard', employee_id=employee['id']))

        return 'Invalid credentials', 401
    return render_template('login.html')

@app.route('/admin/users', methods=['GET', 'POST'])
@login_required
def users_admin():
    if not current_user.is_admin:
        return "Access Denied", 403
    cur = mysql.connection.cursor(DictCursor)
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        role = request.form.get('role', 'member').strip() or 'member'
        if not email or not password:
            return 'Email and password required', 400
        # Check if email already exists
        cur.execute("SELECT id FROM users WHERE email=%s", (email,))
        if cur.fetchone():
            cur.close()
            return 'Email already exists', 400
        is_admin = 1 if role.lower() == "admin" else 0
        cur.execute("INSERT INTO users (email, password, role, is_admin) VALUES (%s, %s, %s, %s)",
                    (email, password, role, is_admin))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('users_admin'))
    cur.execute("SELECT * FROM users ORDER BY id ASC")
    users = cur.fetchall()
    cur.close()
    return render_template('users.html', users=users)

@app.route('/logout')
def logout():
    log_write('logout')
    logout_user()
    return redirect(url_for('login'))

@app.route('/employee/login', methods=['POST'])
def employee_login():
    """Employee login route"""
    try:
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        # employee_id may be provided from the UI but we don't require it for auth
        if not email or not password:
            return jsonify({'success': False, 'error': 'Missing credentials'}), 400

        cur = mysql.connection.cursor(DictCursor)
        # Ensure password column exists; if not, fail with a clear message
        try:
            cur.execute("SHOW COLUMNS FROM employees LIKE 'password'")
            has_password_col = cur.fetchone() is not None
        except Exception:
            has_password_col = False
        if not has_password_col:
            cur.close()
            return jsonify({'success': False, 'error': "employees.password column missing. Run add_password_column.sql or migration."}), 500
        cur = mysql.connection.cursor(DictCursor)
        cur.execute(
            """
            SELECT * FROM employees 
            WHERE email = %s AND password = %s AND is_active = 1
            """,
            (email, password),
        )
        employee = cur.fetchone()
        cur.close()

        if employee:
            # Create a simple session for employee (not using Flask-Login for employees)
            session['employee_id'] = employee['id']
            session['employee_name'] = employee['name']
            session['employee_email'] = employee['email']
            session['employee_department'] = employee['department']

            log_write('employee_login', f"Employee {employee['name']} logged in")
            return jsonify({'success': True, 'employee_id': employee['id']})
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/employee/logout')
def employee_logout():
    """Employee logout route"""
    if 'employee_id' in session:
        log_write('employee_logout', f"Employee {session.get('employee_name', 'Unknown')} logged out")
        session.pop('employee_id', None)
        session.pop('employee_name', None)
        session.pop('employee_email', None)
        session.pop('employee_department', None)
    return redirect(url_for('login'))

# --- Dashboard Routes ---
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for('master_dashboard'))
        elif current_user.is_supervisor:
            return redirect(url_for('supervisor_dashboard'))
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Redirect non-admins to their role-specific dashboards
    if current_user.is_admin:
        return redirect(url_for('master_dashboard'))
    elif current_user.is_supervisor:
        return redirect(url_for('supervisor_dashboard'))
    return redirect(url_for('role_dashboard'))

@app.route('/dashboard/role')
@login_required
def role_dashboard():
    if current_user.is_admin:
        return redirect(url_for('master_dashboard'))
    role = (current_user.role or 'member').strip()
    role_key = role.lower()

    # Map business flow stages to roles
    role_to_stage = {
        'business dev': 'business',
        'business': 'business',  # backward compatibility
        'bdm': 'business',
        'design': 'design',
        'operations': 'operations',
        'site manager': 'site_manager'
    }
    next_stage_map = {
        'analyzer': 'business',
        'business': 'design',
        'design': 'operations',
        'operations': 'site_manager',
        'site_manager': 'handover'
    }

    current_stage_for_role = role_to_stage.get(role_key)
    cur = mysql.connection.cursor(DictCursor)
    bids = []
    # Query by go_bids.state (not bid_assign.depart) and join assignee info
    role_to_stage = {
        'business dev': 'business',
        'business': 'business', 
        'bdm': 'business',
        'design': 'design',
        'operations': 'operations',
        'site manager': 'engineer',
        'site_manager': 'engineer'
    }
    current_stage_for_role = role_to_stage.get((current_user.role or 'member').lower())
    
    sql = """
        SELECT gb.g_id AS id,
               gb.b_name AS name,
               gb.company, 
               gb.due_date,
               COALESCE(gb.scoring, 0) AS progress,
               LOWER(COALESCE(gb.state, 'analyzer')) AS current_stage,
               ba.person_name,
               ba.assignee_email AS person_email,
               gb.summary
        FROM go_bids gb
        LEFT JOIN bid_assign ba ON ba.g_id = gb.g_id
        {where}
        ORDER BY gb.due_date ASC
    """
    where = "WHERE LOWER(COALESCE(gb.state, 'analyzer')) = %s" if current_stage_for_role else ""
    cur.execute(sql.format(where=where), (current_stage_for_role,) if current_stage_for_role else ())
    bids = cur.fetchall()
    # Keep next_stage mapping in Python and pass it to the template
    next_stage_map = {
        'analyzer': 'business',
        'business': 'design', 
        'design': 'operations',
        'operations': 'engineer',
        'engineer': 'handover'
    }
    for bid in bids:
        bid['user'] = {'email': bid.get('person_email'), 'role': current_stage_for_role}
        bid['next_stage'] = next_stage_map.get(bid['current_stage'])
        # Add dynamic progress and status texts
        stage_key = (bid.get('current_stage') or 'analyzer').lower()
        bid['work_progress_pct'] = stage_progress_pct(stage_key)
        bid['project_status'], bid['work_status'] = status_texts(stage_key)
    # Build dynamic stage lists per bid similar to supervisor view
    # Ensure tables exist
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bid_stage_exclusions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            g_id INT NOT NULL,
            stage VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_bid_stage (g_id, stage)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bid_custom_stages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            g_id INT NOT NULL,
            stage VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_custom_stage (g_id, stage)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )
    # Load exclusions and customs
    cur.execute("SELECT g_id, stage FROM bid_stage_exclusions")
    ex_rows = cur.fetchall()
    exclusions = {}
    for r in ex_rows:
        exclusions.setdefault(r['g_id'], set()).add((r['stage'] or '').strip().lower())
    cur.execute("SELECT g_id, stage FROM bid_custom_stages")
    cs_rows = cur.fetchall()
    customs = {}
    for r in cs_rows:
        customs.setdefault(r['g_id'], []).append((r['stage'] or '').strip().lower())
    default_stages = ['analyzer', 'business', 'design', 'operations', 'engineer']
    for bid in bids:
        bid_id = bid.get('id')
        excl = exclusions.get(bid_id, set())
        cust = [s for s in customs.get(bid_id, []) if s not in excl]
        dyn_stages = [s for s in default_stages if s not in excl] + [s for s in cust if s not in default_stages]
        bid['dyn_stages'] = dyn_stages

    cur.close()

    # Use a generic role dashboard that includes an advance-stage button
    normalized_role = ' '.join([w.capitalize() for w in role_key.split()])
    return render_template('dashboard_role.html', bids=bids, user=current_user, role=normalized_role,
                           current_stage=current_stage_for_role, next_stage_map=next_stage_map)

@app.route('/dashboard/<team>')
@login_required
def team_dashboard(team):
    """Team-specific dashboard for Business Dev, Design, Operations, Site Manager"""
    if current_user.is_admin:
        return redirect(url_for('master_dashboard'))
    
    # Map team names to stages
    team_to_stage = {
        'business': 'business',
        'design': 'design', 
        'operations': 'operations',
        'engineer': 'engineer'
    }
    
    if team not in team_to_stage:
        return "Invalid team", 404
    
    current_stage = team_to_stage[team]
    # Ensure seed tasks exist so all bids appear on this team dashboard
    ensure_tasks_for_team(current_stage)
    cur = mysql.connection.cursor(DictCursor)
    
    # Show bids either assigned to this stage OR having tasks for this team's stage
    cur.execute("""
        SELECT gb.g_id AS id,
               gb.b_name AS name,
               gb.company, 
               gb.due_date,
               COALESCE(gb.scoring, 0) AS progress,
               LOWER(COALESCE(gb.state, 'analyzer')) AS current_stage,
               ba.person_name,
               ba.assignee_email AS person_email,
               ba.depart,
               wps.pr_completion_status AS work_status,
               wbr.closure_status AS project_status,
               wbr.work_progress_status AS work_progress_status,
               wlr.result AS wl_result,
               gb.summary
        FROM go_bids gb
        LEFT JOIN bid_assign ba ON ba.g_id = gb.g_id
        LEFT JOIN win_lost_results wlr ON wlr.a_id = ba.a_id
        LEFT JOIN won_bids_result wbr ON wbr.w_id = wlr.w_id
        LEFT JOIN work_progress_status wps ON wps.won_id = wbr.won_id
        WHERE LOWER(COALESCE(gb.state, 'analyzer')) = %s
           OR EXISTS (
               SELECT 1 FROM bid_checklists bc
               WHERE bc.g_id = gb.g_id AND LOWER(COALESCE(bc.stage,'')) = %s
           )
        ORDER BY gb.due_date ASC
    """, (current_stage, current_stage))
    
    bids = cur.fetchall()
    
    # Get team stats from go_bids.state
    cur.execute("""
        SELECT COUNT(*) AS total_bids
        FROM go_bids gb
        WHERE LOWER(COALESCE(gb.state, 'analyzer')) = %s
           OR EXISTS (
               SELECT 1 FROM bid_checklists bc
               WHERE bc.g_id = gb.g_id AND LOWER(COALESCE(bc.stage,'')) = %s
           )
    """, (current_stage, current_stage))
    
    total_bids = cur.fetchone()['total_bids']
    
    cur.execute("""
        SELECT COUNT(*) AS completed_bids
        FROM go_bids gb
        LEFT JOIN bid_assign ba ON ba.g_id = gb.g_id
        LEFT JOIN win_lost_results wlr ON wlr.a_id = ba.a_id
        WHERE (LOWER(COALESCE(gb.state, 'analyzer')) = %s
               OR EXISTS (
                   SELECT 1 FROM bid_checklists bc
                   WHERE bc.g_id = gb.g_id AND LOWER(COALESCE(bc.stage,'')) = %s
               ))
          AND wlr.result = 'WON'
    """, (current_stage, current_stage))
    
    completed_bids = cur.fetchone()['completed_bids']
    
    cur.close()
    
    # Map stage names for display
    stage_display_names = {
        'business': 'Business Development',
        'design': 'Design Team', 
        'operations': 'Operations Team',
        'engineer': 'Site Engineer'
    }
    
    team_display_name = stage_display_names.get(team, team.title())
    
    # Add dynamic progress and status texts to each bid
    for bid in bids:
        stage_key = (bid.get('current_stage') or 'analyzer').lower()
        bid['work_progress_pct'] = stage_progress_pct(stage_key)
        bid['project_status'], bid['work_status'] = status_texts(stage_key)
    
    # Define next stage mapping for template
    def get_next_stage(current_stage):
        stage_flow = {
            'analyzer': 'business',
            'business': 'design',
            'design': 'operations',
            'operations': 'engineer',
            'engineer': 'handover'
        }
        return stage_flow.get(current_stage, None)
    
    return render_template('team_dashboard.html', 
                         bids=bids, 
                         team=team,
                         team_display_name=team_display_name,
                         current_stage=current_stage,
                         total_bids=total_bids,
                         completed_bids=completed_bids,
                         user=current_user,
                         get_next_stage=get_next_stage)

main_stats = {
    'total_bids': 350, 'live_bids': 75, 'bids_won': 120, 'projects_completed': 95,
}

# Assuming you have a Log model, e.g., from SQLAlchemy


# ... other imports and app setup

@app.route('/logs')
@login_required
def logs_page():
    # Query all logs, ordering by the most recent first
    cur = mysql.connection.cursor(DictCursor)
    cur.execute("SELECT * FROM logs ORDER BY timestamp DESC")
    all_logs = cur.fetchall()
    cur.close()
    return render_template('logs.html', logs=all_logs)

@app.route('/supervisor-dashboard')
@login_required
def supervisor_dashboard():
    if not current_user.can_assign_stages:
        return "Access Denied", 403
    
    cur = mysql.connection.cursor(DictCursor)
    # Ensure exclusions table exists
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bid_stage_exclusions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            g_id INT NOT NULL,
            stage VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_bid_stage (g_id, stage)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )

    # Ensure custom stages table exists
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bid_custom_stages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            g_id INT NOT NULL,
            stage VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_custom_stage (g_id, stage)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )

    # Get all GO bids for stage assignment
    cur.execute("""
        SELECT gb.g_id, gb.b_name, gb.state, gb.company, gb.revenue, gb.type,
               ba.person_name, ba.depart
        FROM go_bids gb
        LEFT JOIN bid_assign ba ON ba.g_id = gb.g_id
        ORDER BY gb.g_id DESC
    """)
    go_bids = cur.fetchall()
    # Fetch stage exclusions per bid
    cur.execute("SELECT g_id, stage FROM bid_stage_exclusions")
    rows = cur.fetchall()
    stage_exclusions = {}
    for r in rows:
        stage_exclusions.setdefault(r['g_id'], []).append(r['stage'])

    # Fetch custom stages per bid
    cur.execute("SELECT g_id, stage FROM bid_custom_stages")
    rows = cur.fetchall()
    custom_stages = {}
    for r in rows:
        custom_stages.setdefault(r['g_id'], []).append(r['stage'])

    # Get stage options
    stages = ['analyzer', 'business', 'design', 'operations', 'engineer']

    # Attach dynamic stages and progress to each bid row for table rendering
    default_stages = stages.copy()
    for bid in go_bids:
        bid_id = bid['g_id']
        excluded = set(stage_exclusions.get(bid_id, []))
        customs = [s for s in custom_stages.get(bid_id, []) if s not in excluded]
        dyn_stages = [s for s in default_stages if s not in excluded] + [s for s in customs if s not in default_stages]
        current_stage = (bid.get('state') or 'analyzer').strip().lower()
        if dyn_stages and current_stage in dyn_stages:
            idx = dyn_stages.index(current_stage)
            pct = int(round((idx / (max(1, len(dyn_stages) - 1))) * 100))
        else:
            pct = 0
        bid['dyn_stages'] = dyn_stages
        bid['dyn_progress_pct'] = pct
    
    cur.close()
    
    return render_template('supervisor_dashboard.html', 
                         go_bids=go_bids, 
                         stages=stages,
                         stage_exclusions=stage_exclusions,
                         custom_stages=custom_stages,
                         user=current_user)

@app.route('/supervisor-projects')
@login_required
def supervisor_projects():
    if not current_user.can_assign_stages:
        return "Access Denied", 403
    cur = mysql.connection.cursor(DictCursor)
    cur.execute(
        """
        SELECT wlr.w_id, wlr.a_id, wlr.b_name, wlr.in_date, wlr.due_date,
               wlr.state, wlr.scope, wlr.value, wlr.company, wlr.department,
               wlr.person_name, wlr.status, wlr.result
        FROM win_lost_results wlr
        ORDER BY wlr.w_id DESC
        """
    )
    projects = cur.fetchall()
    cur.close()
    return render_template('supervisor_projects.html', projects=projects, user=current_user)

@app.route('/supervisor/assign-stage', methods=['POST'])
@login_required
def supervisor_assign_stage():
    if not current_user.can_assign_stages:
        return "Access Denied", 403
    
    g_id = request.form.get('g_id')
    new_stage = request.form.get('stage')
    person_name = request.form.get('person_name', '').strip()
    department = request.form.get('department', '').strip()
    
    if not g_id or not new_stage:
        flash('Missing required fields', 'error')
        return redirect(url_for('supervisor_dashboard'))
    
    cur = mysql.connection.cursor(DictCursor)
    try:
        # Update go_bids state
        cur.execute("UPDATE go_bids SET state=%s WHERE g_id=%s", (new_stage, g_id))
        
        # Update or create bid_assign entry
        cur.execute("SELECT a_id FROM bid_assign WHERE g_id=%s", (g_id,))
        row = cur.fetchone()
        
        if row:
            # Update existing assignment
            cur.execute("""
                UPDATE bid_assign 
                SET state=%s, depart=%s, person_name=%s, status='assigned' 
                WHERE g_id=%s
            """, (new_stage, department, person_name, g_id))
        else:
            # Create new assignment
            cur.execute("""
                INSERT INTO bid_assign (g_id, b_name, in_date, due_date, state, scope, type, company, depart, person_name, status)
                SELECT g_id, b_name, in_date, due_date, %s, scope, type, company, %s, %s, 'assigned'
                FROM go_bids WHERE g_id=%s
            """, (new_stage, department, person_name, g_id))
        
        # Log the action
        cur.execute("SELECT b_name FROM go_bids WHERE g_id=%s", (g_id,))
        bid_name = cur.fetchone()['b_name']
        log_action = f"Supervisor '{current_user.email}' assigned bid '{bid_name}' (ID: {g_id}) to {new_stage} stage"
        cur.execute("INSERT INTO logs (action, user_id) VALUES (%s, %s)", (log_action, current_user.id))
        
        mysql.connection.commit()
        flash(f'Bid "{bid_name}" assigned to {new_stage} stage successfully!', 'success')
        
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Assignment failed: {str(e)}', 'error')
    finally:
        cur.close()
    
    return redirect(url_for('supervisor_dashboard'))

@app.route('/team/stage/add', methods=['POST'])
@login_required
def team_add_stage():
    """Allow team members to add a custom stage for a bid without supervisor role.
    The stage is stored in bid_custom_stages and un-excluded if previously excluded.
    """
    g_id = request.form.get('g_id')
    stage = (request.form.get('new_stage') or '').strip().lower()
    if not g_id or not stage:
        return jsonify({'ok': False, 'error': 'Missing parameters'}), 400
    cur = mysql.connection.cursor(DictCursor)
    try:
        # Ensure required tables exist
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bid_stage_exclusions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                g_id INT NOT NULL,
                stage VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_bid_stage (g_id, stage)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bid_custom_stages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                g_id INT NOT NULL,
                stage VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_custom_stage (g_id, stage)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute("INSERT IGNORE INTO bid_custom_stages (g_id, stage) VALUES (%s, %s)", (g_id, stage))
        cur.execute("DELETE FROM bid_stage_exclusions WHERE g_id=%s AND stage=%s", (g_id, stage))
        mysql.connection.commit()
        return jsonify({'ok': True})
    except Exception as e:
        mysql.connection.rollback()
        return jsonify({'ok': False, 'error': str(e)}), 500
    finally:
        cur.close()

@app.route('/team/stage/delete', methods=['POST'])
@login_required
def team_delete_stage():
    """Allow team members to delete a stage that comes after the current stage."""
    g_id = request.form.get('g_id')
    stage = (request.form.get('stage') or '').strip().lower()
    if not g_id or not stage:
        return jsonify({'ok': False, 'error': 'Missing parameters'}), 400
    cur = mysql.connection.cursor(DictCursor)
    try:
        # Ensure required tables exist
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bid_stage_exclusions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                g_id INT NOT NULL,
                stage VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_bid_stage (g_id, stage)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bid_custom_stages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                g_id INT NOT NULL,
                stage VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_custom_stage (g_id, stage)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Determine current stage and dynamic stage ordering
        cur.execute("SELECT state FROM go_bids WHERE g_id=%s", (g_id,))
        row = cur.fetchone()
        current_stage = (row.get('state') or 'analyzer').strip().lower() if row else 'analyzer'

        cur.execute("SELECT stage FROM bid_stage_exclusions WHERE g_id=%s", (g_id,))
        excluded = { (r['stage'] or '').strip().lower() for r in cur.fetchall() }
        cur.execute("SELECT stage FROM bid_custom_stages WHERE g_id=%s", (g_id,))
        customs = [ (r['stage'] or '').strip().lower() for r in cur.fetchall() ]
        default_stages = ['analyzer', 'business', 'design', 'operations', 'engineer']
        stages = [s for s in default_stages if s not in excluded] + [s for s in customs if s not in excluded and s not in default_stages]

        if stage not in stages:
            return jsonify({'ok': False, 'error': 'Stage not found for this bid.'}), 400
        try:
            curr_idx = stages.index(current_stage)
        except ValueError:
            curr_idx = 0
        target_idx = stages.index(stage)
        if target_idx <= curr_idx:
            return jsonify({'ok': False, 'error': 'Cannot delete current or previous stages.'}), 400

        cur.execute("INSERT IGNORE INTO bid_stage_exclusions (g_id, stage) VALUES (%s, %s)", (g_id, stage))
        cur.execute("DELETE FROM bid_custom_stages WHERE g_id=%s AND stage=%s", (g_id, stage))
        mysql.connection.commit()
        return jsonify({'ok': True})
    except Exception as e:
        mysql.connection.rollback()
        return jsonify({'ok': False, 'error': str(e)}), 500
    finally:
        cur.close()

@app.route('/supervisor/delete-stage', methods=['POST'])
@login_required
def supervisor_delete_stage():
    if not current_user.can_assign_stages:
        return "Access Denied", 403
    g_id = request.form.get('g_id')
    deleted_stage = request.form.get('stage')
    if not g_id:
        flash('Missing bid id', 'error')
        return redirect(url_for('supervisor_dashboard'))
    cur = mysql.connection.cursor(DictCursor)
    try:
        # Remove explicit assignment for this bid; keep current go_bids.state unchanged
        cur.execute("DELETE FROM bid_assign WHERE g_id=%s", (g_id,))
        # Persist exclusion so UI can hide this stage next time
        if deleted_stage:
            cur.execute(
                "INSERT IGNORE INTO bid_stage_exclusions (g_id, stage) VALUES (%s, %s)",
                (g_id, deleted_stage)
            )
        # Log action
        cur.execute("SELECT b_name FROM go_bids WHERE g_id=%s", (g_id,))
        row = cur.fetchone()
        bid_name = row['b_name'] if row else g_id
        stage_info = f" (stage: {deleted_stage})" if deleted_stage else ""
        log_action = f"Supervisor '{current_user.email}' deleted stage assignment for bid '{bid_name}' (ID: {g_id}){stage_info}"
        cur.execute("INSERT INTO logs (action, user_id) VALUES (%s, %s)", (log_action, current_user.id))
        mysql.connection.commit()
        flash('Stage assignment deleted.', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Could not delete stage: {e}', 'error')
    finally:
        cur.close()
    return redirect(url_for('supervisor_dashboard'))

@app.route('/supervisor/delete-bid', methods=['POST'])
@login_required
def supervisor_delete_bid():
    if not current_user.can_assign_stages:
        return "Access Denied", 403
    g_id = request.form.get('g_id')
    if not g_id:
        flash('Missing bid id', 'error')
        return redirect(url_for('supervisor_dashboard'))
    cur = mysql.connection.cursor(DictCursor)
    try:
        # Fetch name for logs
        cur.execute("SELECT b_name FROM go_bids WHERE g_id=%s", (g_id,))
        row = cur.fetchone()
        bid_name = (row or {}).get('b_name', g_id)

        # Delete related records to avoid foreign key constraint errors
        cur.execute("DELETE FROM bid_checklists WHERE g_id=%s", (g_id,))
        cur.execute("DELETE FROM bid_stage_exclusions WHERE g_id=%s", (g_id,))
        cur.execute("DELETE FROM bid_custom_stages WHERE g_id=%s", (g_id,))
        
        # Delete from bid_assign and cascading tables
        cur.execute("SELECT a_id FROM bid_assign WHERE g_id=%s", (g_id,))
        assign_ids = [r['a_id'] for r in cur.fetchall()]
        
        for a_id in assign_ids:
            cur.execute("SELECT w_id FROM win_lost_results WHERE a_id=%s", (a_id,))
            w_ids = [r['w_id'] for r in cur.fetchall()]
            
            for w_id in w_ids:
                cur.execute("SELECT won_id FROM won_bids_result WHERE w_id=%s", (w_id,))
                won_ids = [r['won_id'] for r in cur.fetchall()]
                
                for won_id in won_ids:
                    cur.execute("DELETE FROM work_progress_status WHERE won_id=%s", (won_id,))
                
                cur.execute("DELETE FROM won_bids_result WHERE w_id=%s", (w_id,))
            
            cur.execute("DELETE FROM win_lost_results WHERE a_id=%s", (a_id,))
        
        cur.execute("DELETE FROM bid_assign WHERE g_id=%s", (g_id,))
        cur.execute("DELETE FROM go_bids WHERE g_id=%s", (g_id,))

        # Log
        log_action = f"Supervisor '{current_user.email}' deleted bid '{bid_name}' (ID: {g_id})"
        cur.execute("INSERT INTO logs (action, user_id) VALUES (%s, %s)", (log_action, current_user.id))
        mysql.connection.commit()
        flash('Bid deleted successfully.', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Failed to delete bid: {e}', 'error')
    finally:
        cur.close()
    return redirect(url_for('supervisor_dashboard'))

@app.route('/supervisor/set-stage', methods=['POST'])
@login_required
def supervisor_set_stage():
    if not current_user.can_assign_stages:
        return "Access Denied", 403
    g_id = request.form.get('g_id')
    new_stage = (request.form.get('new_stage') or '').strip().lower()
    if not g_id or not new_stage:
        flash('Missing bid id or stage.', 'error')
        return redirect(url_for('supervisor_dashboard'))
    cur = mysql.connection.cursor(DictCursor)
    try:
        # Update go_bids current stage only
        cur.execute("UPDATE go_bids SET state=%s WHERE g_id=%s", (new_stage, g_id))
        # Log
        cur.execute("SELECT b_name FROM go_bids WHERE g_id=%s", (g_id,))
        bid_name = (cur.fetchone() or {}).get('b_name', g_id)
        log_action = f"Supervisor '{current_user.email}' set stage to '{new_stage}' for bid '{bid_name}' (ID: {g_id})"
        cur.execute("INSERT INTO logs (action, user_id) VALUES (%s, %s)", (log_action, current_user.id))
        mysql.connection.commit()
        flash('Stage updated.', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Could not update stage: {e}', 'error')
    finally:
        cur.close()
    return redirect(url_for('supervisor_dashboard'))

@app.route('/supervisor/add-stage', methods=['POST'])
@login_required
def supervisor_add_stage():
    if not current_user.can_assign_stages:
        return "Access Denied", 403
    g_id = request.form.get('g_id')
    stage = (request.form.get('new_stage') or '').strip().lower()
    if not g_id or not stage:
        flash('Bid id and stage name are required.', 'error')
        return redirect(url_for('supervisor_dashboard'))
    cur = mysql.connection.cursor(DictCursor)
    try:
        cur.execute("INSERT IGNORE INTO bid_custom_stages (g_id, stage) VALUES (%s, %s)", (g_id, stage))
        # If stage was previously excluded, un-exclude it
        cur.execute("DELETE FROM bid_stage_exclusions WHERE g_id=%s AND stage=%s", (g_id, stage))
        cur.execute("SELECT b_name FROM go_bids WHERE g_id=%s", (g_id,))
        bid_name = (cur.fetchone() or {}).get('b_name', g_id)
        log_action = f"Supervisor '{current_user.email}' added custom stage '{stage}' for bid '{bid_name}' (ID: {g_id})"
        cur.execute("INSERT INTO logs (action, user_id) VALUES (%s, %s)", (log_action, current_user.id))
        mysql.connection.commit()
        flash('Stage added.', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Could not add stage: {e}', 'error')
    finally:
        cur.close()
    return redirect(url_for('supervisor_dashboard'))

@app.route('/supervisor/remove-stage', methods=['POST'])
@login_required
def supervisor_remove_stage():
    if not current_user.can_assign_stages:
        return "Access Denied", 403
    g_id = request.form.get('g_id')
    stage = (request.form.get('stage') or '').strip().lower()
    if not g_id or not stage:
        return {"ok": False, "error": "Missing parameters"}, 400
    cur = mysql.connection.cursor(DictCursor)
    try:
        # Fetch current stage and dynamic stages for this bid
        cur.execute("SELECT state FROM go_bids WHERE g_id=%s", (g_id,))
        row = cur.fetchone()
        current_stage = (row.get('state') or 'analyzer').strip().lower() if row else 'analyzer'

        # Build dynamic stages order (defaults - exclusions + customs)
        cur.execute("SELECT stage FROM bid_stage_exclusions WHERE g_id=%s", (g_id,))
        excluded = { (r['stage'] or '').strip().lower() for r in cur.fetchall() }
        cur.execute("SELECT stage FROM bid_custom_stages WHERE g_id=%s", (g_id,))
        customs = [ (r['stage'] or '').strip().lower() for r in cur.fetchall() ]
        default_stages = ['analyzer', 'business', 'design', 'operations', 'engineer']
        stages = [s for s in default_stages if s not in excluded] + [s for s in customs if s not in excluded and s not in default_stages]

        if stage not in stages:
            return {"ok": False, "error": "Stage not found for this bid."}, 400
        try:
            curr_idx = stages.index(current_stage)
        except ValueError:
            curr_idx = 0
        target_idx = stages.index(stage)
        if target_idx <= curr_idx:
            return {"ok": False, "error": "Cannot delete current or previous stages."}, 400

        # Mark as excluded and remove from custom if present
        cur.execute("INSERT IGNORE INTO bid_stage_exclusions (g_id, stage) VALUES (%s, %s)", (g_id, stage))
        cur.execute("DELETE FROM bid_custom_stages WHERE g_id=%s AND stage=%s", (g_id, stage))
        mysql.connection.commit()
        return {"ok": True, "message": "Stage deleted."}
    except Exception as e:
        mysql.connection.rollback()
        return {"ok": False, "error": str(e)}, 500
    finally:
        cur.close()

@app.route('/master-dashboard')
@login_required
def master_dashboard():
    if not current_user.is_admin:
        return "Access Denied", 403
    
    cur = mysql.connection.cursor(DictCursor)
    
    # Get all bids from go_bids
    cur.execute(
        """
        SELECT gb.g_id AS id,
               gb.b_name AS name,
               COALESCE(gb.state, 'analyzer') AS current_stage,
               '' AS user_email,
               '' AS user_role
        FROM go_bids gb
        """
    )
    all_bids = cur.fetchall()
    # Attach nested user object for template compatibility (bid.user.email, bid.user.role)
    for bid in all_bids:
        bid['user'] = {
            'email': bid.get('user_email'),
            'role': bid.get('user_role')
        }
    
    # Get all companies
    cur.execute("SELECT * FROM companies")
    companies = cur.fetchall()
    
    # Get top 5 projects from all companies (most urgent due dates) with company name
    cur.execute("""
        SELECT p.*, c.name AS company_name
        FROM projects p
        JOIN companies c ON p.company_id = c.id
        ORDER BY p.due_date ASC
        LIMIT 5
    """)
    top_projects = cur.fetchall()
    # Attach nested company object for template compatibility (project.company.name)
    for project in top_projects:
        project['company'] = {'name': project.get('company_name')}
    has_projects = len(top_projects) > 0
    
    # Get all tasks
    cur.execute("SELECT * FROM tasks")
    all_tasks = cur.fetchall()

    # Compute real-time stats for cards across the three companies only using go_bids
    cur.execute("SELECT name FROM companies WHERE name IN ('Ikio','Metco','Sunsprint')")
    target_company_names = [row['name'] for row in cur.fetchall()]
    if not target_company_names:
        target_company_names = ['__none__']
    in_clause = ','.join(['%s'] * len(target_company_names))
    # Also fetch ids for the projects metrics below
    cur.execute("SELECT id FROM companies WHERE name IN ('Ikio','Metco','Sunsprint')")
    target_company_ids = [row['id'] for row in cur.fetchall()] or [-1]
    in_clause_ids = ','.join(['%s'] * len(target_company_ids))

    cur.execute(f"SELECT COUNT(*) AS total_bids FROM go_bids WHERE company IN ({in_clause})", target_company_names)
    total_bids = cur.fetchone()['total_bids']

    cur.execute(f"""
        SELECT COUNT(*) AS live_bids
        FROM go_bids
        WHERE COALESCE(state,'analyzer') IN ('business','design','operations','engineer')
          AND company IN ({in_clause})
    """, target_company_names)
    live_bids = cur.fetchone()['live_bids']

    cur.execute(f"SELECT COUNT(*) AS bids_won FROM go_bids WHERE decision='WON' AND company IN ({in_clause})", target_company_names)
    bids_won = cur.fetchone()['bids_won']

    # Build per-company stage counts from go_bids for the timeline trackers
    # Per-state counts
    cur.execute(
        """
        SELECT COALESCE(company, '') AS company,
               LOWER(COALESCE(state, '')) AS state,
               COUNT(*) AS total
        FROM go_bids
        GROUP BY COALESCE(company, ''), LOWER(COALESCE(state, ''))
        """
    )
    rows = cur.fetchall()
    go_counts = {}
    for row in rows:
        company_key = row['company'] or ''
        if company_key not in go_counts:
            go_counts[company_key] = {'analyzer': 0, 'business': 0, 'design': 0, 'operations': 0, 'engineer': 0, 'handover': 0}
        # Map known downstream stages directly; others will be reflected in analyzer via totals below
        if row['state'] in ('business', 'design', 'operations', 'engineer', 'handover'):
            go_counts[company_key][row['state']] = row['total']

    # Total per company should power the Analyzer start point
    cur.execute(
        """
        SELECT COALESCE(company, '') AS company,
               COUNT(*) AS total
        FROM go_bids
        GROUP BY COALESCE(company, '')
        """
    )
    totals = cur.fetchall()
    for t in totals:
        company_key = t['company'] or ''
        if company_key not in go_counts:
            go_counts[company_key] = {'analyzer': 0, 'business': 0, 'design': 0, 'operations': 0, 'engineer': 0, 'handover': 0}
        go_counts[company_key]['analyzer'] = t['total']
    company_names = [c['name'] for c in companies]

    # Build go_bids by company for dropdown summaries, enriched with status from downstream tables
    cur.execute(
        """
        SELECT gb.g_id,
               gb.b_name,
               gb.in_date,
               gb.due_date,
               gb.state,
               gb.type,
               gb.company,
               gb.decision,
               gb.summary,
               gb.scoring AS progress,
               wps.pr_completion_status AS work_status,
               wps.dept_bde,
               wps.dept_m_d,
               wps.dept_op,
               wps.dept_site,
               wbr.closure_status AS project_status,
               wbr.work_progress_status AS work_progress_status,
               wlr.result AS wl_result
        FROM go_bids gb
        LEFT JOIN bid_assign ba ON ba.g_id = gb.g_id
        LEFT JOIN win_lost_results wlr ON wlr.a_id = ba.a_id
        LEFT JOIN won_bids_result wbr ON wbr.w_id = wlr.w_id
        LEFT JOIN work_progress_status wps ON wps.won_id = wbr.won_id
        ORDER BY gb.company ASC, gb.due_date ASC
        """
    )
    go_rows = cur.fetchall()
    go_bids_by_company = {}
    for r in go_rows:
        cname = (r.get('company') or '').strip()
        go_bids_by_company.setdefault(cname, []).append(r)
    go_company_names = [name for name, rows in go_bids_by_company.items() if name and len(rows) > 0]

    # assigned_bids handles assignments now; no assign_go_bids refresh

    # Build a flat list of go_bids projects with normalized tracker progress
    def _normalize_stage(raw_state: str) -> str:
        s = (raw_state or '').strip().lower()
        mapping = {
            'analyzer': 'analyzer',
            'business': 'business', 'business dev': 'business', 'bdm': 'business',
            'design': 'design',
            'operations': 'operations', 'operation': 'operations',
            'engineer': 'engineer', 'site_manager': 'engineer', 'site manager': 'engineer',
            'handover': 'handover', 'won': 'handover'
        }
        # Treat unknown values like submitted/under_review/pending as analyzer
        return mapping.get(s, 'analyzer')

    stage_to_percent = {
        'analyzer': 0,
        'business': 20,
        'design': 40,
        'operations': 60,
        'engineer': 80,
        'handover': 100,
    }

    # Dynamic stages (respect supervisor-managed deletions and additions)
    cur.execute("SELECT g_id, stage FROM bid_stage_exclusions")
    ex_rows = cur.fetchall()
    exclusions_by_bid = {}
    for row in ex_rows:
        exclusions_by_bid.setdefault(row['g_id'], set()).add((row['stage'] or '').strip().lower())
    cur.execute("SELECT g_id, stage FROM bid_custom_stages")
    cs_rows = cur.fetchall()
    customs_by_bid = {}
    for row in cs_rows:
        customs_by_bid.setdefault(row['g_id'], []).append((row['stage'] or '').strip().lower())

    default_stages = ['analyzer', 'business', 'design', 'operations', 'engineer', 'handover']

    def _parse_percent(value: str) -> int:
        try:
            if value is None:
                return 0
            s = str(value)
            # extract digits first
            import re
            m = re.search(r"(\d{1,3})", s)
            if m:
                pct = max(0, min(100, int(m.group(1))))
                return pct
            s = s.strip().lower()
            if s in ('done','completed','closed','handover','100%'):
                return 100
            if s in ('in_progress','ongoing'):
                return 50
            if s in ('pending','todo','not_started'):
                return 0
            return 0
        except Exception:
            return 0

    # Helper: compute stage progress map from bid_checklists per bid
    def _compute_stage_map_for_bid(g_id: int) -> dict:
        try:
            cur2 = mysql.connection.cursor(DictCursor)
            cur2.execute("""
                SELECT bc.progress_pct, bc.status, COALESCE(bc.stage, u.role) AS stage_source
                FROM bid_checklists bc
                LEFT JOIN users u ON bc.created_by = u.id
                WHERE bc.g_id = %s
            """, (g_id,))
            rows = cur2.fetchall()
            cur2.close()
            role_to_stage = {
                'business dev': 'business',
                'design': 'design',
                'operations': 'operations',
                'site manager': 'engineer',
                'engineer': 'engineer'
            }
            buckets = {'analyzer': [], 'business': [], 'design': [], 'operations': [], 'engineer': [], 'handover': []}
            for r in rows:
                source = (r.get('stage_source') or '').strip().lower()
                stage = role_to_stage.get(source, source if source in buckets else None)
                if not stage:
                    continue
                pct = r.get('progress_pct')
                if pct is None:
                    s = (r.get('status') or '').strip().lower()
                    pct = 100 if s == 'completed' else 50 if s == 'in_progress' else 0
                try:
                    pct = max(0, min(100, int(pct)))
                except Exception:
                    pct = 0
                buckets[stage].append(pct)
            def avg(lst):
                if not lst:
                    return 0
                return int(round(sum(lst) / len(lst)))
            return {
                'analyzer': avg(buckets['analyzer']),
                'business': avg(buckets['business']),
                'design': avg(buckets['design']),
                'operations': avg(buckets['operations']),
                'engineer': avg(buckets['engineer']),
                'handover': avg(buckets['handover']),
            }
        except Exception:
            return {'analyzer': 0, 'business': 0, 'design': 0, 'operations': 0, 'engineer': 0, 'handover': 0}

    go_projects = []
    for r in go_rows:
        stage_key = _normalize_stage(r.get('state'))
        excluded = exclusions_by_bid.get(r.get('g_id'), set())
        custom = [s for s in customs_by_bid.get(r.get('g_id'), []) if s not in excluded]
        stages = [s for s in default_stages if s not in excluded] + [s for s in custom if s not in default_stages]
        if stages and stage_key in stages:
            idx = stages.index(stage_key)
            stage_progress = int(round((idx / (max(1, len(stages) - 1))) * 100))
        else:
            stage_progress = stage_to_percent.get(stage_key, 0)
        progress_pct = r.get('progress') if r.get('progress') is not None else r.get('scoring')
        try:
            progress_pct = max(0, min(100, int(progress_pct))) if progress_pct is not None else None
        except Exception:
            progress_pct = None
        
        # Calculate dynamic progress and status texts
        item_pct = stage_progress_pct(stage_key)
        proj_status, work_status = status_texts(stage_key)
        
        # Prefer task-based progress; fallback to work_progress_status when tasks are absent
        task_stage_map = _compute_stage_map_for_bid(r.get('g_id'))
        if any(v > 0 for v in task_stage_map.values()):
            stage_progress_map = {
                'analyzer': 0,
                'business': task_stage_map.get('business', 0),
                'design': task_stage_map.get('design', 0),
                'operations': task_stage_map.get('operations', 0),
                'engineer': task_stage_map.get('engineer', 0),
                'handover': 0,
            }
        else:
            stage_progress_map = {
                'analyzer': 0,
                'business': _parse_percent(r.get('dept_bde')),
                'design': _parse_percent(r.get('dept_m_d')),
                'operations': _parse_percent(r.get('dept_op')),
                'engineer': _parse_percent(r.get('dept_site')),
                'handover': 100 if (r.get('project_status') or '').strip().lower() in ('closed','completed','handover','done') else 0,
            }

        go_projects.append({
            'g_id': r.get('g_id'),
            'b_name': r.get('b_name'),
            'company': r.get('company'),
            'state': r.get('state'),
            'stage_key': stage_key,
            'stage_progress': stage_progress,
            'stages': stages,
            'current_stage': stage_key,
            'in_date': r.get('in_date'),
            'due_date': r.get('due_date'),
            'type': r.get('type'),
            'decision': r.get('decision') or r.get('wl_result'),
            'project_status': proj_status,  # New dynamic project status
            'work_status': work_status,     # New dynamic work status
            'summary': r.get('summary'),
            'progress_pct': progress_pct,
            'work_progress_pct': item_pct,  # New dynamic progress
            'stage_progress_map': stage_progress_map,
        })

    # Total projects across target companies (fallback to all projects if none linked)
    cur.execute(f"SELECT COUNT(*) AS projects_linked FROM projects WHERE company_id IN ({in_clause_ids})", target_company_ids)
    projects_linked = cur.fetchone()['projects_linked']
    if projects_linked > 0:
        cur.execute(f"SELECT COUNT(*) AS projects_total FROM projects WHERE company_id IN ({in_clause_ids})", target_company_ids)
        projects_total = cur.fetchone()['projects_total']
    else:
        cur.execute("SELECT COUNT(*) AS projects_total FROM projects")
        projects_total = cur.fetchone()['projects_total']

    # Project-based dashboard cards (map to existing template keys)
    # total_bids -> Total Projects
    # live_bids -> Projects Completed
    # bids_won -> Projects (in progress)
    # projects_completed -> Projects (on hold)
    if projects_linked > 0:
        cur.execute(f"SELECT COUNT(*) AS projects_completed FROM projects WHERE status='completed' AND company_id IN ({in_clause_ids})", target_company_ids)
        p_completed = cur.fetchone()['projects_completed']
        cur.execute(f"SELECT COUNT(*) AS projects_in_progress FROM projects WHERE status IN ('active','in_progress') AND company_id IN ({in_clause_ids})", target_company_ids)
        p_in_progress = cur.fetchone()['projects_in_progress']
        cur.execute(f"SELECT COUNT(*) AS projects_on_hold FROM projects WHERE status='on_hold' AND company_id IN ({in_clause_ids})", target_company_ids)
        p_on_hold = cur.fetchone()['projects_on_hold']
    else:
        cur.execute("SELECT COUNT(*) AS projects_completed FROM projects WHERE status='completed'")
        p_completed = cur.fetchone()['projects_completed']
        cur.execute("SELECT COUNT(*) AS projects_in_progress FROM projects WHERE status IN ('active','in_progress')")
        p_in_progress = cur.fetchone()['projects_in_progress']
        cur.execute("SELECT COUNT(*) AS projects_on_hold FROM projects WHERE status='on_hold'")
        p_on_hold = cur.fetchone()['projects_on_hold']

    # Override dashboard cards using win_lost_results as requested
    try:
        cur.execute("SELECT COUNT(*) AS total_projects_wlr FROM win_lost_results")
        total_projects_wlr = cur.fetchone()['total_projects_wlr']
    except Exception:
        total_projects_wlr = projects_total

    try:
        cur.execute("""
            SELECT COUNT(*) AS in_progress_wlr
            FROM win_lost_results
            WHERE COALESCE(UPPER(result), '') NOT IN ('WON','LOST')
        """)
        in_progress_wlr = cur.fetchone()['in_progress_wlr']
    except Exception:
        in_progress_wlr = p_in_progress

    data = {
        'total_bids': total_projects_wlr,      # Total Projects
        'live_bids': p_completed,               # Projects Completed (unchanged)
        'bids_won': in_progress_wlr,            # Projects (in progress)
        'projects_completed': p_on_hold         # Projects (on hold)
    }

    # Compute bid analyzer stats from bid_incoming table
    cur.execute("SELECT COUNT(*) AS total_bids FROM bid_incoming")
    total_bids_analyzer = cur.fetchone()['total_bids']
    
    cur.execute("SELECT COUNT(*) AS bids_go FROM bid_incoming WHERE decision = 'GO'")
    bids_go_analyzer = cur.fetchone()['bids_go']
    
    cur.execute("SELECT COUNT(*) AS bids_no_go FROM bid_incoming WHERE decision = 'NO-GO'")
    bids_no_go_analyzer = cur.fetchone()['bids_no_go']
    
    cur.execute("SELECT COUNT(*) AS bids_submitted FROM bid_incoming WHERE state IN ('submitted', 'under_review')")
    bids_submitted_analyzer = cur.fetchone()['bids_submitted']
    
    cur.execute("SELECT COUNT(*) AS bids_won FROM bid_incoming WHERE decision = 'WON'")
    bids_won_analyzer = cur.fetchone()['bids_won']
    
    cur.execute("SELECT COUNT(*) AS bids_lost FROM bid_incoming WHERE decision = 'LOST'")
    bids_lost_analyzer = cur.fetchone()['bids_lost']

    bid_stats = {
        'total_bids': total_bids_analyzer,
        'bids_go': bids_go_analyzer,
        'bids_no_go': bids_no_go_analyzer,
        'bids_submitted': bids_submitted_analyzer,
        'bids_won': bids_won_analyzer,
        'bids_lost': bids_lost_analyzer
    }
    
    # Ensure won_bids_result contains all WON decisions from bid_incoming
    cur.execute(
        """
        INSERT INTO won_bids_result (w_id)
        SELECT DISTINCT wlr.w_id
        FROM bid_incoming bi
        LEFT JOIN go_bids gb ON gb.b_id = bi.b_id
        LEFT JOIN bid_assign ba ON ba.g_id = gb.g_id
        LEFT JOIN win_lost_results wlr ON wlr.a_id = ba.a_id
        LEFT JOIN won_bids_result wbr ON wbr.w_id = wlr.w_id
        WHERE UPPER(bi.decision) = 'WON'
          AND wlr.w_id IS NOT NULL
          AND wbr.w_id IS NULL
        """
    )
    mysql.connection.commit()

    # Build Won Projects (latest 5) timeline data
    # Ensure won_bids_result has rows for any WIN in win_lost_results
    cur.execute(
        """
        INSERT INTO won_bids_result (w_id)
        SELECT wlr.w_id
        FROM win_lost_results wlr
        LEFT JOIN won_bids_result wbr ON wbr.w_id = wlr.w_id
        WHERE UPPER(COALESCE(wlr.result,'')) = 'WON'
          AND wbr.w_id IS NULL
        """
    )
    mysql.connection.commit()

    cur.execute(
        """
        SELECT wbr.*, gb.g_id, gb.b_name, gb.company,
               COALESCE(wps.pr_completion_status, wbr.work_progress_status) AS work_progress_status
        FROM won_bids_result wbr
        LEFT JOIN win_lost_results wlr ON wlr.w_id = wbr.w_id
        LEFT JOIN bid_assign ba ON ba.a_id = wlr.a_id
        LEFT JOIN go_bids gb ON gb.g_id = ba.g_id
        LEFT JOIN work_progress_status wps ON wps.won_id = wbr.won_id
        ORDER BY wbr.w_id DESC
        LIMIT 5
        """
    )
    won_rows = cur.fetchall()

    def _won_stage_key(row: dict) -> str:
        closure = (row.get('closure_status') or '').strip().lower()
        work_prog = (row.get('work_progress_status') or '').strip().lower()
        if closure in ('closed', 'completed', 'handover', 'done'):
            return 'closure'
        if work_prog in ('operations', 'operation', 'ops', 'in_progress'):
            return 'operations'
        if work_prog == 'design':
            return 'design'
        return 'business'

    won_stages_default = ['won', 'business', 'design', 'operations', 'closure']
    won_projects = []
    for r in won_rows:
        stage_key = _won_stage_key(r)
        idx = won_stages_default.index(stage_key) if stage_key in won_stages_default else 0
        stage_progress = int(round((idx / (len(won_stages_default) - 1)) * 100)) if len(won_stages_default) > 1 else 0
        won_projects.append({
            'g_id': r.get('g_id'),
            'b_name': r.get('b_name') or f"Won #{r.get('won_id')}",
            'company': r.get('company'),
            'current_stage': stage_key,
            'stage_progress': stage_progress,
            'stages': won_stages_default,
            'stage_progress_map': {
                'won': 100 if stage_key == 'won' else 0,
                'business': 100 if stage_key == 'business' else 0,
                'design': 100 if stage_key == 'design' else 0,
                'operations': 100 if stage_key == 'operations' else 0,
                'closure': 100 if stage_key == 'closure' else 0,
            }
        })

    # Fallback: include directly WON bids from bid_incoming not present above
    cur.execute("""
        SELECT bi.b_id, bi.b_name, COALESCE(gb.company,'') AS company
        FROM bid_incoming bi
        LEFT JOIN go_bids gb ON gb.b_id = bi.b_id
        WHERE UPPER(bi.decision) = 'WON'
        ORDER BY bi.b_id DESC
        LIMIT 20
    """)
    bi_won = cur.fetchall() or []
    existing_names = { (p['b_name'] or '').strip() for p in won_projects }
    for r in bi_won:
        name = r.get('b_name')
        if name and name.strip() in existing_names:
            continue
        won_projects.append({
            'g_id': r.get('b_id'),
            'b_name': name,
            'company': r.get('company'),
            'current_stage': 'business',
            'stage_progress': int(round((won_stages_default.index('business')/(len(won_stages_default)-1))*100)),
            'stages': won_stages_default,
            'stage_progress_map': {
                'won': 0,
                'business': 100,
                'design': 0,
                'operations': 0,
                'closure': 0,
            }
        })

    # Last fallback: if still empty, use most recent go_bids as pseudo-projects
    if not won_projects:
        cur.execute("SELECT g_id, b_name, company FROM go_bids ORDER BY COALESCE(due_date, NOW()) DESC LIMIT 5")
        gb_rows = cur.fetchall() or []
        for r in gb_rows:
            won_projects.append({
                'g_id': r.get('g_id'),
                'b_name': r.get('b_name'),
                'company': r.get('company'),
                'current_stage': 'business',
                'stage_progress': int(round((won_stages_default.index('business')/(len(won_stages_default)-1))*100)),
                'stages': won_stages_default,
                'stage_progress_map': {
                    'won': 0,
                    'business': 0,
                    'design': 0,
                    'operations': 0,
                    'closure': 0,
                }
            })

    # Split go_projects into top 5 by due_date and the rest
    def _date_key(item):
        try:
            return item.get('due_date') or ''
        except Exception:
            return ''
    go_projects_sorted = sorted(go_projects, key=_date_key)
    go_projects_top5 = go_projects_sorted[:5]
    go_projects_more = go_projects_sorted[5:]

    # Parallel kickoff: ensure every bid appears in each team dashboard
    for team_key in ['business','design','operations','engineer']:
        ensure_tasks_for_team(team_key)

    cur.close()

    return render_template(
        'master_dashboard.html', 
        page='dashboard',
        title='Master Dashboard',
        data=data,
        bids=all_bids,
        companies=companies,
        projects=top_projects,
        tasks=all_tasks,
        bid_stats=bid_stats,
        go_counts=go_counts,
        has_projects=has_projects,
        company_names=company_names,
        go_company_names=go_company_names,
        go_bids_by_company=go_bids_by_company,
        go_projects_top5=go_projects_top5,
        go_projects_more=go_projects_more,
        won_projects=won_projects
    )

@app.route('/company/<company_name>')
@login_required
def company_dashboard(company_name):
    if not current_user.is_admin:
        return "Access Denied", 403
    
    # Map URL names to database names
    company_mapping = {
        'ikio': 'Ikio',
        'metco': 'Metco', 
        'sunsprint': 'Sunsprint'
    }
    
    db_company_name = company_mapping.get(company_name.lower())
    if not db_company_name:
        return "Company not found", 404
        
    cur = mysql.connection.cursor(DictCursor)
    
    # Get company
    cur.execute("SELECT * FROM companies WHERE name=%s", (db_company_name,))
    company = cur.fetchone()
    if not company:
        cur.close()
        return "Company not found", 404
    
    # Get top 5 projects for this company with nested company info
    cur.execute("""
        SELECT p.*, c.name AS company_name
        FROM projects p
        JOIN companies c ON p.company_id = c.id
        WHERE p.company_id=%s
        ORDER BY p.due_date ASC
        LIMIT 5
    """, (company['id'],))
    projects = cur.fetchall()
    for project in projects:
        project['company'] = {'name': project.get('company_name')}
    
    # Get tasks for this company
    cur.execute("""
        SELECT t.* FROM tasks t 
        JOIN projects p ON t.project_id = p.id 
        WHERE p.company_id = %s
    """, (company['id'],))
    tasks = cur.fetchall()
    
    # Helper: compute per-stage progress map (average progress_pct per stage for a bid)
    def compute_stage_progress_map_local(bid_id: int) -> dict:
        cur2 = mysql.connection.cursor(DictCursor)
        try:
            stage_keys = ['business','design','operations','engineer']
            stage_map = {}
            for st in stage_keys:
                cur2.execute("SELECT progress_pct, status FROM bid_checklists WHERE g_id=%s AND LOWER(COALESCE(stage,''))=%s", (bid_id, st))
                rows = cur2.fetchall()
                if not rows:
                    stage_map[st] = 0
                    continue
                vals = []
                for r in rows:
                    pct = r.get('progress_pct')
                    if pct is None:
                        s = (r.get('status') or '').lower()
                        pct = 100 if s == 'completed' else 50 if s == 'in_progress' else 0
                    try:
                        vals.append(max(0, min(100, int(pct))))
                    except Exception:
                        vals.append(0)
                stage_map[st] = int(round(sum(vals) / max(1, len(vals))))
            return stage_map
        finally:
            cur2.close()

    # Get bids for this company from go_bids by company name (enriched)
    cur.execute("""
        SELECT gb.*, 
               wps.pr_completion_status AS work_status,
               wbr.closure_status AS project_status,
               wbr.work_progress_status AS work_progress_status,
               wlr.result AS wl_result
        FROM go_bids gb
        LEFT JOIN bid_assign ba ON ba.g_id = gb.g_id
        LEFT JOIN win_lost_results wlr ON wlr.a_id = ba.a_id
        LEFT JOIN won_bids_result wbr ON wbr.w_id = wlr.w_id
        LEFT JOIN work_progress_status wps ON wps.won_id = wbr.won_id
        WHERE gb.company = %s
        ORDER BY gb.due_date ASC
    """, (company['name'],))
    bids = cur.fetchall()
    
    # Company-wise metrics (cards)
    cur.execute("SELECT COUNT(*) AS c FROM go_bids WHERE company=%s", (company['name'],))
    total_bids_company = (cur.fetchone() or {}).get('c', 0)
    cur.execute("""
        SELECT COUNT(*) AS c
        FROM bid_checklists bc
        JOIN go_bids gb ON gb.g_id = bc.g_id
        WHERE gb.company=%s
    """, (company['name'],))
    total_tasks_company = (cur.fetchone() or {}).get('c', 0)
    cur.execute("""
        SELECT COUNT(*) AS c
        FROM win_lost_results wlr
        JOIN bid_assign ba ON ba.a_id = wlr.a_id
        JOIN go_bids gb ON gb.g_id = ba.g_id
        WHERE gb.company=%s AND UPPER(COALESCE(wlr.result,''))='WON'
    """, (company['name'],))
    completed_projects_company = (cur.fetchone() or {}).get('c', 0)
    cur.execute("""
        SELECT AVG(CASE WHEN bc.progress_pct IS NULL THEN 
                 CASE LOWER(COALESCE(bc.status,'')) WHEN 'completed' THEN 100 WHEN 'in_progress' THEN 50 ELSE 0 END
            ELSE bc.progress_pct END) AS avg_pct
        FROM bid_checklists bc
        JOIN go_bids gb ON gb.g_id = bc.g_id
        WHERE gb.company=%s
    """, (company['name'],))
    avg_progress_company = int(round((cur.fetchone() or {}).get('avg_pct') or 0))
    company_metrics = {
        'active_projects': total_bids_company,
        'total_tasks': total_tasks_company,
        'completed_projects': completed_projects_company,
        'avg_progress_pct': avg_progress_company,
    }

    # Fetch stage exclusions and custom stages for each bid
    cur.execute("SELECT g_id, stage FROM bid_stage_exclusions")
    rows = cur.fetchall()
    stage_exclusions = {}
    for r in rows:
        stage_exclusions.setdefault(r['g_id'], []).append(r['stage'])

    cur.execute("SELECT g_id, stage FROM bid_custom_stages")
    rows = cur.fetchall()
    custom_stages = {}
    for r in rows:
        custom_stages.setdefault(r['g_id'], []).append(r['stage'])

    # Prepare go_projects for this company similar to master dashboard
    def _normalize_stage_company(raw_state: str) -> str:
        s = (raw_state or '').strip().lower()
        mapping = {
            'analyzer': 'analyzer',
            'business': 'business', 'business dev': 'business', 'bdm': 'business',
            'design': 'design',
            'operations': 'operations', 'operation': 'operations',
            'engineer': 'engineer', 'site_manager': 'engineer', 'site manager': 'engineer',
            'handover': 'handover', 'won': 'handover'
        }
        return mapping.get(s, 'analyzer')

    stage_to_percent_company = {
        'analyzer': 0,
        'business': 20,
        'design': 40,
        'operations': 60,
        'engineer': 80,
        'handover': 100,
    }

    # Default stages list
    default_stages = ['analyzer', 'business', 'design', 'operations', 'engineer']
    
    go_projects = []
    for r in bids:
        bid_id = r.get('g_id')
        
        # Get dynamic stages for this bid
        excluded = set(stage_exclusions.get(bid_id, []))
        customs = [s for s in custom_stages.get(bid_id, []) if s not in excluded]
        stages = [s for s in default_stages if s not in excluded] + [s for s in customs if s not in default_stages]
        
        # If no stages remain, fall back to default
        if not stages:
            stages = default_stages.copy()
        
        stage_key = _normalize_stage_company(r.get('state'))
        
        # Calculate stage_progress based on dynamic stages
        if stage_key in stages:
            idx = stages.index(stage_key)
            stage_progress = int(round((idx / max(1, len(stages) - 1)) * 100))
        else:
            stage_progress = 0
        
        progress_pct = r.get('scoring')
        try:
            progress_pct = max(0, min(100, int(progress_pct))) if progress_pct is not None else None
        except Exception:
            progress_pct = None
        
        # Calculate dynamic progress and status texts
        item_pct = stage_progress_pct(stage_key)
        proj_status, work_status = status_texts(stage_key)
        
        go_projects.append({
            'g_id': bid_id,
            'b_name': r.get('b_name'),
            'company': r.get('company'),
            'state': r.get('state'),
            'stage_key': stage_key,
            'stage_progress': stage_progress,
            'stages': stages,  # Add dynamic stages list
            'in_date': r.get('in_date'),
            'due_date': r.get('due_date'),
            'type': r.get('type'),
            'decision': r.get('decision') or r.get('wl_result'),
            'project_status': proj_status,  # New dynamic project status
            'work_status': work_status,     # New dynamic work status
            'summary': r.get('summary'),
            'progress_pct': progress_pct,
            'work_progress_pct': item_pct,  # New dynamic progress
            'stage_progress_map': compute_stage_progress_map_local(bid_id),
        })
    
    # Build company-specific Won Projects timeline data (like master, filtered by company)
    cur = mysql.connection.cursor(DictCursor)
    cur.execute(
        """
        INSERT INTO won_bids_result (w_id)
        SELECT wlr.w_id
        FROM win_lost_results wlr
        LEFT JOIN won_bids_result wbr ON wbr.w_id = wlr.w_id
        WHERE UPPER(COALESCE(wlr.result,'')) = 'WON'
          AND wbr.w_id IS NULL
        """
    )
    mysql.connection.commit()

    cur.execute(
        """
        SELECT wbr.*, gb.g_id, gb.b_name, gb.company,
               COALESCE(wps.pr_completion_status, wbr.work_progress_status) AS work_progress_status
        FROM won_bids_result wbr
        LEFT JOIN win_lost_results wlr ON wlr.w_id = wbr.w_id
        LEFT JOIN bid_assign ba ON ba.a_id = wlr.a_id
        LEFT JOIN go_bids gb ON gb.g_id = ba.g_id
        LEFT JOIN work_progress_status wps ON wps.won_id = wbr.won_id
        WHERE gb.company = %s
        ORDER BY wbr.w_id DESC
        LIMIT 5
        """,
        (company['name'],)
    )
    won_rows = cur.fetchall() or []

    def _won_stage_key(row: dict) -> str:
        closure = (row.get('closure_status') or '').strip().lower()
        work_prog = (row.get('work_progress_status') or '').strip().lower()
        if closure in ('closed', 'completed', 'handover', 'done', 'closure'):
            return 'closure'
        if work_prog in ('operations', 'operation', 'ops', 'in_progress'):
            return 'operations'
        if work_prog == 'design':
            return 'design'
        return 'business'

    won_stages_default = ['won', 'business', 'design', 'operations', 'closure']
    won_projects = []
    for r in won_rows:
        stage_key = _won_stage_key(r)
        idx = won_stages_default.index(stage_key) if stage_key in won_stages_default else 0
        stage_progress = int(round((idx / (len(won_stages_default) - 1)) * 100)) if len(won_stages_default) > 1 else 0
        won_projects.append({
            'g_id': r.get('g_id'),
            'b_name': r.get('b_name') or f"Won #{r.get('won_id')}",
            'company': r.get('company'),
            'current_stage': stage_key,
            'stage_progress': stage_progress,
            'stages': won_stages_default,
            # simple per-stage map: mark current stage percentage as stage_progress, others 0
            'stage_progress_map': {
                'won': 100 if stage_key == 'won' else 0,
                'business': 100 if stage_key == 'business' else 0,
                'design': 100 if stage_key == 'design' else 0,
                'operations': 100 if stage_key == 'operations' else 0,
                'closure': 100 if stage_key == 'closure' else 0,
            }
        })

    # Fallbacks for company Projects Timeline so it's never empty
    if not won_projects:
        # Fallback 1: direct WON decisions from bid_incoming for this company
        cur.execute("""
            SELECT bi.b_id AS g_id, bi.b_name, COALESCE(gb.company,'') AS company
            FROM bid_incoming bi
            LEFT JOIN go_bids gb ON gb.b_id = bi.b_id
            WHERE UPPER(bi.decision) = 'WON' AND COALESCE(gb.company,'') = %s
            ORDER BY bi.b_id DESC
            LIMIT 10
        """, (company['name'],))
        bi_rows = cur.fetchall() or []
        for r in bi_rows:
            won_projects.append({
                'g_id': r.get('g_id'),
                'b_name': r.get('b_name'),
                'company': r.get('company'),
                'current_stage': 'business',
                'stage_progress': int(round((won_stages_default.index('business')/(len(won_stages_default)-1))*100)),
                'stages': won_stages_default,
                'stage_progress_map': {
                    'won': 0,
                    'business': 100,
                    'design': 0,
                    'operations': 0,
                    'closure': 0,
                }
            })

    if not won_projects:
        # Fallback 2: most recent company go_bids as pseudo projects
        cur.execute("SELECT g_id, b_name, company FROM go_bids WHERE company=%s ORDER BY COALESCE(due_date, NOW()) DESC LIMIT 5", (company['name'],))
        gb_rows = cur.fetchall() or []
        for r in gb_rows:
            won_projects.append({
                'g_id': r.get('g_id'),
                'b_name': r.get('b_name'),
                'company': r.get('company'),
                'current_stage': 'business',
                'stage_progress': int(round((won_stages_default.index('business')/(len(won_stages_default)-1))*100)),
                'stages': won_stages_default,
                'stage_progress_map': {
                    'won': 0,
                    'business': 0,
                    'design': 0,
                    'operations': 0,
                    'closure': 0,
                }
            })

    cur.close()

    return render_template(
        'company_dashboard.html',
        company=company,
        projects=projects,
        tasks=tasks,
        bids=bids,
        go_projects=go_projects,
        won_projects=won_projects,
        company_metrics=company_metrics,
    )

@app.route('/bid-analyzer')
@login_required
def bid_analyzer():
    if not current_user.is_admin:
        return "Access Denied", 403
    
    cur = mysql.connection.cursor(DictCursor)
    
    # Get bid_incoming data for the table
    cur.execute("SELECT * FROM bid_incoming ORDER BY b_id DESC")
    bid_incoming_data = cur.fetchall()
    
    # Calculate bid stats from bid_incoming table
    cur.execute("SELECT COUNT(*) AS total_bids FROM bid_incoming")
    total_bids = cur.fetchone()['total_bids']
    
    cur.execute("SELECT COUNT(*) AS bids_go FROM bid_incoming WHERE decision = 'GO'")
    bids_go = cur.fetchone()['bids_go']
    
    cur.execute("SELECT COUNT(*) AS bids_no_go FROM bid_incoming WHERE decision = 'NO-GO'")
    bids_no_go = cur.fetchone()['bids_no_go']
    
    cur.execute("SELECT COUNT(*) AS bids_submitted FROM bid_incoming WHERE state IN ('submitted', 'under_review')")
    bids_submitted = cur.fetchone()['bids_submitted']
    
    cur.execute("SELECT COUNT(*) AS bids_won FROM bid_incoming WHERE decision = 'WON'")
    bids_won = cur.fetchone()['bids_won']
    
    cur.execute("SELECT COUNT(*) AS bids_lost FROM bid_incoming WHERE decision = 'LOST'")
    bids_lost = cur.fetchone()['bids_lost']
    
    cur.close()

    return render_template('bid_analyzer_landing.html', bid_cards={
        'total_bids': total_bids,
        'bids_go': bids_go,
        'bids_no_go': bids_no_go,
        'bids_submitted': bids_submitted,
        'bids_won': bids_won,
        'bids_lost': bids_lost
    }, bid_incoming_data=bid_incoming_data)

@app.route('/databases')
@login_required
def databases():
    if not current_user.is_admin:
        return redirect(url_for('role_dashboard'))
    
    # Get all companies and their projects
    cur = mysql.connection.cursor(DictCursor)
    cur.execute("SELECT * FROM companies")
    companies = cur.fetchall()
    company_data = {}
    
    for company in companies:
        cur.execute("SELECT * FROM projects WHERE company_id=%s", (company['id'],))
        projects = cur.fetchall()
        company_data[company['name']] = {
            'company': company,
            'projects': projects,
            'total_revenue': sum(project['revenue'] or 0 for project in projects)
        }
    
    cur.close()
    return render_template('databases.html', company_data=company_data)

@app.route('/databases/create_project', methods=['POST'])
@login_required
def create_project():
    if not current_user.is_admin:
        return redirect(url_for('role_dashboard'))
    
    try:
        company_name = request.form.get('company_name')
        project_name = request.form.get('project_name')
        due_date = request.form.get('due_date')
        revenue = float(request.form.get('revenue', 0))
        status = request.form.get('status', 'active')
        progress = int(request.form.get('progress', 0))
        
        cur = mysql.connection.cursor()
        
        # Find or create company
        cur.execute("SELECT id FROM companies WHERE name=%s", (company_name,))
        company = cur.fetchone()
        if not company:
            cur.execute("INSERT INTO companies (name, description) VALUES (%s, %s)", 
                       (company_name, f'{company_name} Projects'))
            company_id = cur.lastrowid
        else:
            company_id = company['id']
        
        # Create project
        cur.execute("""
            INSERT INTO projects (name, company_id, start_date, due_date, revenue, status, progress) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (project_name, company_id, datetime.utcnow(), 
              datetime.strptime(due_date, '%Y-%m-%d'), revenue, status, progress))
        
        mysql.connection.commit()
        cur.close()
        
        flash(f'Project "{project_name}" created successfully!', 'success')
        
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        flash(f'Error creating project: {str(e)}', 'error')
    
    return redirect(url_for('databases'))

@app.route('/databases/update_project/<int:project_id>', methods=['POST'])
@login_required
def update_project(project_id):
    if not current_user.is_admin:
        return redirect(url_for('role_dashboard'))
    
    try:
        cur = mysql.connection.cursor()
        
        # Check if project exists
        cur.execute("SELECT * FROM projects WHERE id=%s", (project_id,))
        project = cur.fetchone()
        if not project:
            cur.close()
            return "Project not found", 404
        
        # Update project
        project_name = request.form.get('project_name', project['name'])
        due_date = datetime.strptime(request.form.get('due_date'), '%Y-%m-%d')
        revenue = float(request.form.get('revenue', project['revenue'] or 0))
        status = request.form.get('status', project['status'])
        progress = int(request.form.get('progress', project['progress']))
        
        cur.execute("""
            UPDATE projects 
            SET name=%s, due_date=%s, revenue=%s, status=%s, progress=%s 
            WHERE id=%s
        """, (project_name, due_date, revenue, status, progress, project_id))
        
        mysql.connection.commit()
        cur.close()
        
        flash(f'Project "{project_name}" updated successfully!', 'success')
        
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        flash(f'Error updating project: {str(e)}', 'error')
    
    return redirect(url_for('databases'))

@app.route('/databases/delete_project/<int:project_id>')
@login_required
def delete_project(project_id):
    if not current_user.is_admin:
        return redirect(url_for('role_dashboard'))
    
    try:
        cur = mysql.connection.cursor()
        
        # Get project name before deletion
        cur.execute("SELECT name FROM projects WHERE id=%s", (project_id,))
        project = cur.fetchone()
        if not project:
            cur.close()
            return "Project not found", 404
        
        project_name = project['name']
        
        # Delete project
        cur.execute("DELETE FROM projects WHERE id=%s", (project_id,))
        mysql.connection.commit()
        cur.close()
        
        flash(f'Project "{project_name}" deleted successfully!', 'success')
        
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        flash(f'Error deleting project: {str(e)}', 'error')
    
    return redirect(url_for('databases'))

@app.route('/databases/import_excel', methods=['POST'])
@login_required
def import_excel():
    if not current_user.is_admin:
        return redirect(url_for('role_dashboard'))
    
    try:
        if 'excel_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('databases'))
        
        file = request.files['excel_file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('databases'))
        
        if file and file.filename.endswith('.xlsx'):
            import pandas as pd
            import io
            
            # Read Excel file
            df = pd.read_excel(io.BytesIO(file.read()))
            
            # Expected columns: Company, Project Name, Due Date, Revenue, Status, Progress
            required_columns = ['Company', 'Project Name', 'Due Date', 'Revenue', 'Status', 'Progress']
            
            if not all(col in df.columns for col in required_columns):
                flash(f'Excel file must contain columns: {", ".join(required_columns)}', 'error')
                return redirect(url_for('databases'))
            
            projects_created = 0
            cur = mysql.connection.cursor()
            
            for _, row in df.iterrows():
                try:
                    # Find or create company
                    company_name = str(row['Company']).strip()
                    cur.execute("SELECT id FROM companies WHERE name=%s", (company_name,))
                    company = cur.fetchone()
                    if not company:
                        cur.execute("INSERT INTO companies (name, description) VALUES (%s, %s)", 
                                   (company_name, f'{company_name} Projects'))
                        company_id = cur.lastrowid
                    else:
                        company_id = company['id']
                    
                    # Create project
                    cur.execute("""
                        INSERT INTO projects (name, company_id, start_date, due_date, revenue, status, progress) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (str(row['Project Name']).strip(), company_id, datetime.utcnow(),
                          pd.to_datetime(row['Due Date']), 
                          float(row['Revenue']) if pd.notna(row['Revenue']) else 0,
                          str(row['Status']).strip() if pd.notna(row['Status']) else 'active',
                          int(row['Progress']) if pd.notna(row['Progress']) else 0))
                    
                    projects_created += 1
                    
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            mysql.connection.commit()
            cur.close()
            flash(f'Successfully imported {projects_created} projects from Excel file!', 'success')
            
        else:
            flash('Please upload a valid Excel file (.xlsx)', 'error')
            
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        flash(f'Error importing Excel file: {str(e)}', 'error')
    
    return redirect(url_for('databases'))

@app.route('/databases/drop_all')
@login_required
def drop_all_databases():
    if not current_user.is_admin:
        return redirect(url_for('role_dashboard'))
    
    try:
        cur = mysql.connection.cursor()
        
        # Delete all projects
        cur.execute("DELETE FROM projects")
        
        # Delete all companies
        cur.execute("DELETE FROM companies")
        
        # Delete all tasks
        cur.execute("DELETE FROM tasks")
        
        mysql.connection.commit()
        cur.close()
        flash('All databases have been cleared successfully!', 'success')
        
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        flash(f'Error clearing databases: {str(e)}', 'error')
    
    return redirect(url_for('databases'))

# --- Bid Incoming CRUD Routes ---
@app.route('/bid-analyzer/create', methods=['POST'])
@login_required
def create_bid_incoming():
    if not current_user.is_admin:
        return "Access Denied", 403
    
    try:
        cur = mysql.connection.cursor()
        
        b_name = request.form.get('b_name', '').strip()
        in_date = request.form.get('in_date', '').strip()
        due_date = request.form.get('due_date', '').strip()
        state = request.form.get('state', '').strip()
        scope = request.form.get('scope', '').strip()
        type_val = request.form.get('type', '').strip()
        scoring = int(request.form.get('scoring', 0)) if request.form.get('scoring') else None
        comp_name = request.form.get('comp_name', '').strip()
        decision = request.form.get('decision', '').strip()
        summary = request.form.get('summary', '').strip()
        
        if not b_name or not in_date or not due_date:
            return 'Bid name, incoming date and due date are required', 400
        
        cur.execute("""
            INSERT INTO bid_incoming (b_name, in_date, due_date, state, scope, type, scoring, comp_name, decision, summary)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (b_name, in_date, due_date, state, scope, type_val, scoring, comp_name, decision, summary))
        
        bid_id = cur.lastrowid
        mysql.connection.commit()
        
        # Auto-assign GO bids to Business stage
        if (decision or '').upper() == 'GO':
            cur2 = mysql.connection.cursor(DictCursor)
            cur2.execute("SELECT g_id FROM go_bids WHERE b_id=%s", (bid_id,))
            row = cur2.fetchone()
            args = (b_name, in_date, due_date, 'business', scope, type_val, scoring, comp_name, decision, summary)
            if row:
                cur2.execute("""UPDATE go_bids SET b_name=%s,in_date=%s,due_date=%s,state=%s,scope=%s,
                                type=%s,scoring=%s,company=%s,decision=%s,summary=%s WHERE g_id=%s""", (*args, row['g_id']))
            else:
                cur2.execute("""INSERT INTO go_bids (b_id,b_name,in_date,due_date,state,scope,type,scoring,company,decision,summary)
                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                             (bid_id, *args))
            mysql.connection.commit()
            cur2.close()
            log_write('assign', f"Auto GO → Business for bid '{b_name}'")
        
        cur.close()
        log_write('create', f"table=bid_incoming, id={bid_id}")
        flash(f'Bid "{b_name}" created successfully!', 'success')
        
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        flash(f'Error creating bid: {str(e)}', 'error')
    
    return redirect(url_for('bid_analyzer'))

@app.route('/bid-analyzer/update/<int:bid_id>', methods=['POST'])
@login_required
def update_bid_incoming(bid_id):
    if not current_user.is_admin:
        return "Access Denied", 403
    
    try:
        cur = mysql.connection.cursor()
        
        # Check if bid exists
        cur.execute("SELECT * FROM bid_incoming WHERE b_id=%s", (bid_id,))
        bid = cur.fetchone()
        if not bid:
            cur.close()
            return "Bid not found", 404
        
        b_name = request.form.get('b_name', bid['b_name']).strip()
        in_date = request.form.get('in_date', str(bid['in_date']) if bid['in_date'] else '').strip()
        due_date = request.form.get('due_date', str(bid['due_date'])).strip()
        state = request.form.get('state', bid['state'] or '').strip()
        scope = request.form.get('scope', bid['scope'] or '').strip()
        type_val = request.form.get('type', bid['type'] or '').strip()
        scoring = int(request.form.get('scoring', 0)) if request.form.get('scoring') else bid['scoring']
        comp_name = request.form.get('comp_name', bid['comp_name'] or '').strip()
        decision = request.form.get('decision', bid['decision'] or '').strip()
        summary = request.form.get('summary', bid['summary'] or '').strip()
        
        cur.execute("""
            UPDATE bid_incoming 
            SET b_name=%s, in_date=%s, due_date=%s, state=%s, scope=%s, type=%s, scoring=%s, comp_name=%s, decision=%s, summary=%s
            WHERE b_id=%s
        """, (b_name, in_date, due_date, state, scope, type_val, scoring, comp_name, decision, summary, bid_id))
        
        mysql.connection.commit()
        
        # Auto-assign GO bids to Business stage
        if (decision or '').upper() == 'GO':
            cur2 = mysql.connection.cursor(DictCursor)
            cur2.execute("SELECT g_id FROM go_bids WHERE b_id=%s", (bid_id,))
            row = cur2.fetchone()
            args = (b_name, in_date, due_date, 'business', scope, type_val, scoring, comp_name, decision, summary)
            if row:
                cur2.execute("""UPDATE go_bids SET b_name=%s,in_date=%s,due_date=%s,state=%s,scope=%s,
                                type=%s,scoring=%s,company=%s,decision=%s,summary=%s WHERE g_id=%s""", (*args, row['g_id']))
            else:
                cur2.execute("""INSERT INTO go_bids (b_id,b_name,in_date,due_date,state,scope,type,scoring,company,decision,summary)
                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                             (bid_id, *args))
            mysql.connection.commit()
            cur2.close()
            log_write('assign', f"Auto GO → Business for bid '{b_name}'")
        
        cur.close()
        log_write('update', f"table=bid_incoming, id={bid_id}")
        flash(f'Bid "{b_name}" updated successfully!', 'success')
        
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        flash(f'Error updating bid: {str(e)}', 'error')
    
    return redirect(url_for('bid_analyzer'))

@app.route('/bid-analyzer/delete/<int:bid_id>')
@login_required
def delete_bid_incoming(bid_id):
    if not current_user.is_admin:
        return "Access Denied", 403
    
    try:
        cur = mysql.connection.cursor()
        
        # Get bid name before deletion
        cur.execute("SELECT b_name FROM bid_incoming WHERE b_id=%s", (bid_id,))
        bid = cur.fetchone()
        if not bid:
            cur.close()
            return "Bid not found", 404
        
        bid_name = bid['b_name']
        
        # Delete bid
        cur.execute("DELETE FROM bid_incoming WHERE b_id=%s", (bid_id,))
        mysql.connection.commit()
        cur.close()
        
        log_write('delete', f"table=bid_incoming, id={bid_id}")
        flash(f'Bid "{bid_name}" deleted successfully!', 'success')
        
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        flash(f'Error deleting bid: {str(e)}', 'error')
    
    return redirect(url_for('bid_analyzer'))

@app.route('/database-management')
@login_required
def database_management():
    if not current_user.is_admin:
        return "Access Denied", 403
    
    cur = mysql.connection.cursor(DictCursor)
    
    # Get all available tables
    cur.execute("SHOW TABLES")
    all_tables = [list(table.values())[0] for table in cur.fetchall()]
    
    # Filter to show only specific tables in proper sequence
    allowed_tables = ['bid_incoming', 'go_bids', 'bid_assign', 'win_lost_results', 'won_bids_result', 'work_progress_status']
    # Maintain the sequence by preserving the order in allowed_tables
    tables = [t for t in allowed_tables if t in all_tables]
    
    # Create mapping for display names (capital letters with spaces)
    table_display_names = {
        'bid_incoming': 'BID INCOMING',
        'go_bids': 'GO BIDS',
        'bid_assign': 'BID ASSIGN',
        'win_lost_results': 'WIN LOST RESULTS',
        'won_bids_result': 'WON BIDS RESULT',
        'work_progress_status': 'WORK PROGRESS STATUS'
    }
    
    # Function to format column names (e.g., 'b_id' -> 'BID ID', 'comp_name' -> 'COMP NAME')
    def format_column_name(col_name):
        # Replace underscores with spaces and convert to uppercase
        formatted = col_name.replace('_', ' ').upper()
        return formatted
    
    # Get selected table (default to first table)
    selected_table = request.args.get('table', tables[0] if tables else '')
    search_query = request.args.get('search', '')
    
    # Get table data
    table_data = []
    table_columns = []
    companies = []
    
    if selected_table:
        # Get table structure
        cur.execute(f"DESCRIBE `{selected_table}`")
        table_columns = cur.fetchall()
        
        # Auto-sync GO bids into go_bids table
        if selected_table == 'go_bids':
            cur.execute("""
                INSERT INTO go_bids (b_id, b_name, in_date, due_date, state, scope, type, scoring, company, decision, summary)
                SELECT bi.b_id, bi.b_name, bi.in_date, bi.due_date, bi.state, bi.scope, bi.type, bi.scoring, bi.comp_name, bi.decision, bi.summary
                FROM bid_incoming bi
                LEFT JOIN go_bids gb ON gb.b_id = bi.b_id
                WHERE UPPER(bi.decision) = 'GO' AND gb.b_id IS NULL
            """)
            mysql.connection.commit()
            # After syncing GO bids, refresh revenue-based assignments into assigned_bids
            try:
                assign_bids_by_revenue()
            except Exception:
                pass

        # Note: bid_assign is now populated ONLY via explicit Assign action from go_bids

        # Auto-sync bid_assign into win_lost_results (one row per assignment)
        if selected_table == 'win_lost_results':
            cur.execute("""
                INSERT INTO win_lost_results (a_id, b_name, in_date, due_date, state, scope, value, company, department, person_name, status, result)
                SELECT ba.a_id, ba.b_name, ba.in_date, ba.due_date, ba.state, ba.scope, ba.value, ba.company, ba.depart, ba.person_name, ba.status, NULL
                FROM bid_assign ba
                LEFT JOIN win_lost_results wlr ON wlr.a_id = ba.a_id
                WHERE wlr.a_id IS NULL
            """)
            mysql.connection.commit()

        # Auto-sync win_lost_results into won_bids_result (link by w_id)
        if selected_table == 'won_bids_result':
            cur.execute("""
                INSERT INTO won_bids_result (w_id)
                SELECT wlr.w_id
                FROM win_lost_results wlr
                LEFT JOIN won_bids_result wbr ON wbr.w_id = wlr.w_id
                WHERE wbr.w_id IS NULL
            """)
            mysql.connection.commit()

        # Auto-sync won_bids_result into work_progress_status (link by won_id)
        if selected_table == 'work_progress_status':
            cur.execute("""
                INSERT INTO work_progress_status (won_id, company, b_name, dept_bde, dept_m_d, dept_op, dept_site, pr_completion_status)
                SELECT wbr.won_id, COALESCE(gb.company, ''), COALESCE(gb.b_name, ''), '', '', '', '', NULL
                FROM won_bids_result wbr
                LEFT JOIN win_lost_results wlr ON wlr.w_id = wbr.w_id
                LEFT JOIN bid_assign ba ON ba.a_id = wlr.a_id
                LEFT JOIN go_bids gb ON gb.g_id = ba.g_id
                LEFT JOIN work_progress_status wps ON wps.won_id = wbr.won_id
                WHERE wps.won_id IS NULL
            """)
            mysql.connection.commit()

        # Get table data with search
        if search_query:
            # Build search query for all text columns
            search_conditions = []
            for col in table_columns:
                if col['Type'].startswith(('varchar', 'text', 'char')):
                    search_conditions.append(f"`{col['Field']}` LIKE %s")
            
            if search_conditions:
                search_pattern = f"%{search_query}%"
                search_params = [search_pattern] * len(search_conditions)
                where_clause = " OR ".join(search_conditions)
                cur.execute(f"SELECT * FROM `{selected_table}` WHERE {where_clause}", search_params)
            else:
                cur.execute(f"SELECT * FROM `{selected_table}`")
        else:
            cur.execute(f"SELECT * FROM `{selected_table}`")
        
        table_data = cur.fetchall()
        if selected_table == 'bid_incoming':
            cur.execute("SELECT id, name FROM companies ORDER BY name")
            companies = cur.fetchall()
    
    cur.close()
    
    return render_template('database_management.html', 
                         tables=tables, 
                         selected_table=selected_table,
                         table_columns=table_columns,
                         table_data=table_data,
                         search_query=search_query,
                         companies=companies,
                         table_display_names=table_display_names,
                         format_column_name=format_column_name)

@app.route('/admin/refresh-assign-go')
@login_required
def admin_refresh_assign_go():
    if not current_user.is_admin:
        return "Access Denied", 403
    try:
        assign_bids_by_revenue()
        flash('assigned_bids refreshed from go_bids.', 'success')
    except Exception as e:
        flash(f'Failed to refresh: {e}', 'error')
    return redirect(url_for('database_management', table='assigned_bids'))

@app.route('/database-management/create', methods=['POST'])
@login_required
def dbm_create():
    if not current_user.is_admin:
        return "Access Denied", 403
    table = request.form.get('table')
    if not table:
        return redirect(url_for('database_management'))
    cur = mysql.connection.cursor(DictCursor)
    try:
        # Get columns and build insert
        cur.execute(f"DESCRIBE `{table}`")
        cols = cur.fetchall()
        fields = []
        values = []
        params = []
        for c in cols:
            if c['Key'] == 'PRI':
                continue
            field = c['Field']
            fields.append(f"`{field}`")
            values.append('%s')
            params.append(request.form.get(field))
        sql = f"INSERT INTO `{table}` ({', '.join(fields)}) VALUES ({', '.join(values)})"
        cur.execute(sql, params)
        mysql.connection.commit()

        # If a bid was created in bid_incoming, offer auto-assign path via flash info
        if table == 'bid_incoming':
            flash('Bid created in Bid Incoming. Assign it to a company from the actions column.', 'success')
    finally:
        cur.close()
    return redirect(url_for('database_management', table=table))

@app.route('/database-management/update/<int:row_id>', methods=['POST'])
@login_required
def dbm_update(row_id):
    if not current_user.is_admin:
        return "Access Denied", 403
    table = request.form.get('table')
    if not table:
        return redirect(url_for('database_management'))
    cur = mysql.connection.cursor(DictCursor)
    try:
        cur.execute(f"DESCRIBE `{table}`")
        cols = cur.fetchall()
        pk = cols[0]['Field']
        sets = []
        params = []
        for c in cols:
            if c['Key'] == 'PRI':
                pk = c['Field']
                continue
            field = c['Field']
            sets.append(f"`{field}`=%s")
            params.append(request.form.get(field))
        params.append(row_id)
        sql = f"UPDATE `{table}` SET {', '.join(sets)} WHERE `{pk}`=%s"
        cur.execute(sql, params)
        mysql.connection.commit()
    finally:
        cur.close()
    return redirect(url_for('database_management', table=table))

@app.route('/database-management/assign', methods=['POST'])
@login_required
def dbm_assign():
    if not current_user.is_admin:
        return "Access Denied", 403
    bid_incoming_id = request.form.get('bid_id')
    company_id = request.form.get('company_id')
    if not bid_incoming_id or not company_id:
        return redirect(url_for('database_management', table='bid_incoming'))
    cur = mysql.connection.cursor(DictCursor)
    try:
        # Read bid_incoming
        cur.execute("SELECT * FROM bid_incoming WHERE b_id=%s", (bid_incoming_id,))
        inc = cur.fetchone()
        if not inc:
            cur.close()
            return redirect(url_for('database_management', table='bid_incoming'))
        # Create bid in main bids with analyzer -> business based on decision
        stage = 'analyzer'
        if (inc.get('decision') or '').upper() == 'GO':
            stage = 'business'
        cur.execute("INSERT INTO bids (name, current_stage, user_id, company_id) VALUES (%s,%s,%s,%s)",
                    (inc.get('b_name'), stage, current_user.id, company_id))
        new_bid_id = cur.lastrowid
        # Timeline
        cur.execute("INSERT INTO bid_timeline (bid_id, event, details) VALUES (%s,%s,%s)",
                    (new_bid_id, 'assigned', f"Assigned to company_id={company_id} from bid_incoming {bid_incoming_id}"))
        # Optional: keep in bid_incoming or delete; here we keep it
        mysql.connection.commit()
        # Emit via socket if needed later; for now just flash
        flash('Bid assigned to company successfully.', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Assignment failed: {str(e)}', 'error')
    finally:
        cur.close()
    return redirect(url_for('database_management', table='bid_incoming'))

@app.route('/database-management/assign-go', methods=['POST'])
@login_required
def dbm_assign_go():
    if not current_user.is_admin:
        return "Access Denied", 403
    g_id = request.form.get('g_id')
    depart = request.form.get('depart', '').strip()
    person_name = request.form.get('person_name', '').strip()
    email_to = request.form.get('email', '').strip()
    if not g_id or not depart or not person_name:
        flash('Please provide department and person name', 'error')
        return redirect(url_for('database_management', table='go_bids'))
    cur = mysql.connection.cursor(DictCursor)
    try:
        # Ensure bid_assign exists for this go bid; insert if missing else update
        cur.execute("SELECT a_id FROM bid_assign WHERE g_id=%s", (g_id,))
        row = cur.fetchone()
        if not row:
            # Pull from go_bids to seed
            cur.execute("SELECT * FROM go_bids WHERE g_id=%s", (g_id,))
            gb = cur.fetchone()
            if not gb:
                flash('GO bid not found', 'error')
                cur.close()
                return redirect(url_for('database_management', table='go_bids'))
            
            # Map department to stage for go_bids.state update
            dept_to_stage = {
                'business dev': 'business',
                'business': 'business',
                'design': 'design',
                'operations': 'operations',
                'site manager': 'engineer',
                'engineer': 'engineer'
            }
            new_stage = dept_to_stage.get(depart.lower(), depart.lower())
            
            # Update go_bids.state to match assigned department
            cur.execute("UPDATE go_bids SET state=%s WHERE g_id=%s", (new_stage, g_id))
            
            # Check if assignee_email column exists before inserting
            try:
                cur.execute("""
                    INSERT INTO bid_assign (g_id, b_name, in_date, due_date, state, scope, type, company, depart, person_name, assignee_email, status, value, revenue)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'assigned',%s,%s)
                """, (
                    gb['g_id'], gb['b_name'], gb['in_date'], gb['due_date'], new_stage, gb['scope'], gb['type'], gb['company'], depart, person_name, email_to,
                    gb.get('scoring', 0), gb.get('revenue', gb.get('scoring', 0))
                ))
            except Exception as e:
                if "Unknown column 'assignee_email'" in str(e):
                    # Fallback insert without assignee_email
                    cur.execute("""
                        INSERT INTO bid_assign (g_id, b_name, in_date, due_date, state, scope, type, company, depart, person_name, status, value, revenue)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'assigned',%s,%s)
                    """, (
                        gb['g_id'], gb['b_name'], gb['in_date'], gb['due_date'], new_stage, gb['scope'], gb['type'], gb['company'], depart, person_name,
                        gb.get('scoring', 0), gb.get('revenue', gb.get('scoring', 0))
                    ))
                else:
                    raise e
        else:
            # Map department to stage for go_bids.state update
            dept_to_stage = {
                'business dev': 'business',
                'business': 'business',
                'design': 'design',
                'operations': 'operations',
                'site manager': 'engineer',
                'engineer': 'engineer'
            }
            new_stage = dept_to_stage.get(depart.lower(), depart.lower())
            
            # Update go_bids.state to match assigned department
            cur.execute("UPDATE go_bids SET state=%s WHERE g_id=%s", (new_stage, g_id))
            
            # Check if assignee_email column exists before updating
            try:
                cur.execute("UPDATE bid_assign SET depart=%s, person_name=%s, assignee_email=%s, state=%s, status='assigned' WHERE g_id=%s", 
                           (depart, person_name, email_to, new_stage, g_id))
            except Exception as e:
                if "Unknown column 'assignee_email'" in str(e):
                    # Fallback update without assignee_email
                    cur.execute("UPDATE bid_assign SET depart=%s, person_name=%s, state=%s, status='assigned' WHERE g_id=%s", 
                               (depart, person_name, new_stage, g_id))
                else:
                    raise e
        # Log the assignment action
        cur.execute("SELECT b_name FROM go_bids WHERE g_id=%s", (g_id,))
        bid_name = cur.fetchone()['b_name']
        log_action = f"Admin '{current_user.email}' assigned bid '{bid_name}' (ID: {g_id}) to {depart} department - {person_name} ({email_to})"
        cur.execute("INSERT INTO logs (action, user_id) VALUES (%s, %s)", (log_action, current_user.id))
        
        mysql.connection.commit()
        
        # Emit Socket.IO update for real-time master dashboard sync
        socketio.emit('master_update', {
            'bid': {
                'id': g_id,
                'name': bid_name,
                'current_stage': new_stage,
                'assigned_to': person_name,
                'department': depart
            },
            'log': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'action': log_action,
                'user_email': current_user.email,
                'user_role': 'admin'
            },
            'assignment': True
        })
        
        flash('Bid assigned to person successfully.', 'success')
        # Send email notification if provided
        if email_to:
            try:
                send_assignment_email(email_to, gb['b_name'] if 'gb' in locals() else None, depart, person_name, gb['company'] if 'gb' in locals() else None)
            except Exception as _e:
                # Non-fatal
                flash('Assignment email could not be sent.', 'error')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Assignment failed: {str(e)}', 'error')
    finally:
        cur.close()
    return redirect(url_for('database_management', table='go_bids'))

# --- Email helper (adapted from mail_send.py) ---
SMTP_PORT = 587
SMTP_SERVER = "smtp.gmail.com"
EMAIL_FROM = "manuj@metcoengineering.com"
EMAIL_PASSWORD = "ksec iqja bmdg hrcv"

def send_assignment_email(email_to: str, bid_name: str, depart: str, person_name: str, company: str):
    subject = f"Bid Assignment: {bid_name or ''}"
    body = f"You have been assigned to bid '{bid_name or ''}' for company '{company or ''}'.\nDepartment: {depart}\nAssignee: {person_name}\n\nPlease log in to the ESCO suite to proceed."
    message = f"Subject: {subject}\n\n{body}"
    context = ssl.create_default_context()
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    try:
        server.starttls(context=context)
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, email_to, message)
    finally:
        try:
            server.quit()
        except Exception:
            pass

# --- Employee Management Routes ---
@app.route('/team/<team>/employees')
@login_required
def team_employees(team):
    """Manage employees for a specific team"""
    if current_user.is_admin:
        return redirect(url_for('master_dashboard'))
    
    # Map team names to stages
    team_to_stage = {
        'business': 'business',
        'design': 'design', 
        'operations': 'operations',
        'engineer': 'engineer'
    }
    
    if team not in team_to_stage:
        return "Invalid team", 404
    
    cur = mysql.connection.cursor(DictCursor)
    
    # Get employees for this team
    cur.execute("""
        SELECT e.*, u.email as team_lead_email
        FROM employees e
        LEFT JOIN users u ON e.team_lead_id = u.id
        WHERE e.department = %s AND e.is_active = TRUE
        ORDER BY e.name
    """, (team,))
    employees = cur.fetchall()
    
    # Get team leads (users with this role). If none, default to the logged-in admin/lead for convenience.
    cur.execute("SELECT * FROM users WHERE role = %s", (team,))
    team_leads = cur.fetchall()
    if not team_leads and hasattr(current_user, 'id'):
        # Provide a single default option using the current logged-in user
        team_leads = [{'id': current_user.id, 'email': getattr(current_user, 'email', 'current@user')}] 
    
    cur.close()
    
    return render_template('team_employees.html', 
                         team=team, 
                         employees=employees, 
                         team_leads=team_leads,
                         user=current_user)

@app.route('/team/<team>/employees/create', methods=['POST'])
@login_required
def create_employee(team):
    """Create a new employee for the team"""
    try:
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        team_lead_id = request.form.get('team_lead_id')
        
        if not name or not email or not password:
            flash('Name, email, and password are required', 'error')
            return redirect(url_for('team_employees', team=team))
        
        cur = mysql.connection.cursor()
        # Auto-create employee and also create a mapping so add-task dropdown shows both BDE and Site Manager users
        cur.execute("""
            INSERT INTO employees (name, email, password, department, team_lead_id) 
            VALUES (%s, %s, %s, %s, %s)
        """, (name, email, password, team, team_lead_id if team_lead_id else None))
        
        mysql.connection.commit()
        cur.close()
        
        log_write('create_employee', f"Created employee {name} for {team} team")
        flash(f'Employee "{name}" created successfully!', 'success')
        
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        flash(f'Error creating employee: {str(e)}', 'error')
    
    return redirect(url_for('team_employees', team=team))

@app.route('/employee/<int:employee_id>/dashboard')
def employee_dashboard(employee_id):
    """Employee-specific dashboard showing assigned tasks"""
    # Check if employee is logged in and matches the requested employee
    if 'employee_id' not in session or session['employee_id'] != employee_id:
        return "Access denied. Please login first.", 403
    
    cur = mysql.connection.cursor(DictCursor)
    
    # Get employee info
    cur.execute("SELECT * FROM employees WHERE id = %s", (employee_id,))
    employee = cur.fetchone()
    
    if not employee:
        cur.close()
        return "Employee not found", 404
    
    # Get assigned tasks for this employee
    cur.execute("""
        SELECT bc.*, gb.b_name, gb.company, gb.due_date as bid_due_date
        FROM bid_checklists bc
        JOIN go_bids gb ON bc.g_id = gb.g_id
        WHERE bc.assigned_to = %s
        ORDER BY bc.due_date ASC, bc.priority DESC
    """, (employee_id,))
    tasks = cur.fetchall()
    
    # Calculate task statistics
    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks if t['status'] == 'completed'])
    pending_tasks = len([t for t in tasks if t['status'] == 'pending'])
    in_progress_tasks = len([t for t in tasks if t['status'] == 'in_progress'])
    
    cur.close()
    
    return render_template('employee_dashboard.html',
                         employee=employee,
                         tasks=tasks,
                         total_tasks=total_tasks,
                         completed_tasks=completed_tasks,
                         pending_tasks=pending_tasks,
                         in_progress_tasks=in_progress_tasks,
                         user=current_user)

@app.route('/task/<int:task_id>/update_status', methods=['POST'])
def update_task_status(task_id):
    """Update task status and progress"""
    try:
        new_status = request.form.get('status', '').strip()
        progress_notes = request.form.get('progress_notes', '').strip()
        
        if not new_status:
            return jsonify({'error': 'Status is required'}), 400
        
        cur = mysql.connection.cursor(DictCursor)
        
        # Get task info
        cur.execute("""
            SELECT bc.*, gb.b_name, gb.company, e.name as employee_name
            FROM bid_checklists bc
            JOIN go_bids gb ON bc.g_id = gb.g_id
            JOIN employees e ON bc.assigned_to = e.id
            WHERE bc.id = %s
        """, (task_id,))
        task = cur.fetchone()
        
        if not task:
            cur.close()
            return jsonify({'error': 'Task not found'}), 404
        
        # Authorization: allow admins/managers (flask-login) OR the assigned employee via session
        is_flask_user = hasattr(current_user, 'is_authenticated') and current_user.is_authenticated
        employee_id = session.get('employee_id')
        if not is_flask_user:
            # When not logged in via Flask-Login, require employee session and ownership of the task
            if not employee_id or task.get('assigned_to') != employee_id:
                cur.close()
                return jsonify({'error': 'Forbidden'}), 403

        # Update task status (and map to a percentage for persistence)
        status_lower = (new_status or '').strip().lower()
        if status_lower == 'completed':
            pct_val = 100
        elif status_lower == 'in_progress':
            pct_val = 50
        else:
            pct_val = 0
        cur.execute("""
            UPDATE bid_checklists 
            SET status = %s, progress_pct = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (new_status, pct_val, task_id))
        
        # Log the update
        log_write('task_update', 
                 f"Task '{task['task_name']}' for bid '{task['b_name']}' updated to {new_status} by {task['employee_name']}")
        
        mysql.connection.commit()
        cur.close()
        
        # Emit real-time update to team dashboard
        socketio.emit('task_update', {
            'task_id': task_id,
            'status': new_status,
            'bid_name': task['b_name'],
            'employee_name': task['employee_name'],
            'company': task['company']
        })

        # Also emit master_update with per-stage progress for this bid
        try:
            bid_id = task['g_id']
            cur2 = mysql.connection.cursor(DictCursor)
            # Compute per-team completion rates by tasks
            def pct_for(role_expr):
                cur2.execute(f"SELECT status FROM bid_checklists bc JOIN users u ON bc.created_by=u.id WHERE bc.g_id=%s AND u.role {role_expr}", (bid_id,))
                rows = cur2.fetchall()
                if not rows:
                    return 0
                done = len([r for r in rows if (r.get('status') or '').lower()=='completed'])
                return int(round((done/max(1,len(rows)))*100))
            stage_progress_map = {
                'business': pct_for("='business dev'"),
                'design': pct_for("='design'"),
                'operations': pct_for("='operations'"),
                'engineer': pct_for("IN ('site manager','engineer')"),
            }
            cur2.close()
            socketio.emit('master_update', {
                'summary': {
                    'bid_id': bid_id,
                    'work_progress_pct': stage_progress_map.get('design',0),
                    'project_status': 'ongoing',
                    'work_status': f"Task '{task['task_name']}' -> {new_status}",
                    'stage_progress_map': stage_progress_map
                }
            })
        except Exception:
            pass
        
        return jsonify({'success': 'Task status updated successfully'})
        
    except Exception as e:
        mysql.connection.rollback()
        if 'cur' in locals():
            cur.close()
        return jsonify({'error': f'Error updating task: {str(e)}'}), 500

@app.route('/task/<int:task_id>/update', methods=['POST'])
def update_task(task_id):
    """Update task fields (name, description, status, due_date, priority, assigned_to)."""
    try:
        cur = mysql.connection.cursor(DictCursor)
        # Fetch existing task
        cur.execute("SELECT * FROM bid_checklists WHERE id=%s", (task_id,))
        task = cur.fetchone()
        if not task:
            cur.close()
            return jsonify({'error': 'Task not found'}), 404

        # Authorization: admin/manager or assigned employee via session
        is_flask_user = hasattr(current_user, 'is_authenticated') and current_user.is_authenticated
        employee_id = session.get('employee_id')
        if not is_flask_user and (not employee_id or task.get('assigned_to') != employee_id):
            cur.close()
            return jsonify({'error': 'Forbidden'}), 403

        # Collect fields (optional updates)
        fields = []
        values = []
        m = request.form
        if 'task_name' in m:
            fields.append('task_name=%s'); values.append(m.get('task_name').strip())
        if 'description' in m:
            fields.append('description=%s'); values.append(m.get('description').strip())
        if 'status' in m:
            status_val = (m.get('status') or '').strip()
            fields.append('status=%s'); values.append(status_val)
            # Auto-map progress when explicit progress_pct is not provided
            if 'progress_pct' not in m:
                status_lower = status_val.lower()
                if status_lower == 'completed':
                    auto_pct = 100
                elif status_lower == 'in_progress':
                    auto_pct = 50
                else:
                    auto_pct = 0
                fields.append('progress_pct=%s'); values.append(auto_pct)
        if 'due_date' in m:
            fields.append('due_date=%s'); values.append(m.get('due_date'))
        if 'priority' in m:
            fields.append('priority=%s'); values.append(m.get('priority').strip())
        if 'assigned_to' in m:
            at = m.get('assigned_to') or None
            fields.append('assigned_to=%s'); values.append(at)
        # Ensure the task is visible on both Business Dev and Site Manager dashboards when created from either
        if 'stage' in m:
            st = (m.get('stage') or '').strip().lower()
            if st:
                fields.append('stage=%s'); values.append(st)
        if 'progress_pct' in m:
            try:
                pp = int(m.get('progress_pct'))
                pp = max(0, min(100, pp))
            except Exception:
                pp = 0
            fields.append('progress_pct=%s'); values.append(pp)
        if not fields:
            cur.close()
            return jsonify({'error': 'No fields to update'}), 400
        set_clause = ', '.join(fields) + ', updated_at = CURRENT_TIMESTAMP'
        values.append(task_id)
        cur.execute(f"UPDATE bid_checklists SET {set_clause} WHERE id=%s", tuple(values))
        mysql.connection.commit()
        cur.close()
        log_write('task_update_fields', f"task_id={task_id}")
        return jsonify({'success': 'Task updated'})
    except Exception as e:
        mysql.connection.rollback()
        if 'cur' in locals():
            cur.close()
        return jsonify({'error': f'Error updating task: {str(e)}'}), 500

@app.route('/task/<int:task_id>/delete', methods=['POST'])
def delete_task(task_id):
    """Delete a task if authorized."""
    try:
        cur = mysql.connection.cursor(DictCursor)
        cur.execute("SELECT assigned_to FROM bid_checklists WHERE id=%s", (task_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            return jsonify({'error': 'Task not found'}), 404
        is_flask_user = hasattr(current_user, 'is_authenticated') and current_user.is_authenticated
        employee_id = session.get('employee_id')
        if not is_flask_user and (not employee_id or row.get('assigned_to') != employee_id):
            cur.close()
            return jsonify({'error': 'Forbidden'}), 403
        cur.execute("DELETE FROM bid_checklists WHERE id=%s", (task_id,))
        mysql.connection.commit()
        cur.close()
        log_write('task_delete', f"task_id={task_id}")
        return jsonify({'success': 'Task deleted'})
    except Exception as e:
        mysql.connection.rollback()
        if 'cur' in locals():
            cur.close()
        return jsonify({'error': f'Error deleting task: {str(e)}'}), 500

@app.route('/team/<team>/bids/<int:g_id>/checklist')
@login_required
def bid_checklist(team, g_id):
    """Manage checklist/tasks for a specific bid"""
    cur = mysql.connection.cursor(DictCursor)
    
    # Get bid info
    cur.execute("SELECT * FROM go_bids WHERE g_id = %s", (g_id,))
    bid = cur.fetchone()
    
    if not bid:
        cur.close()
        return "Bid not found", 404
    
    # Get checklist items for this bid
    cur.execute("""
        SELECT bc.*, e.name as assigned_employee_name
        FROM bid_checklists bc
        LEFT JOIN employees e ON bc.assigned_to = e.id
        WHERE bc.g_id = %s
        ORDER BY bc.priority DESC, bc.created_at ASC
    """, (g_id,))
    checklist_items = cur.fetchall()
    
    # Get team employees for assignment
    cur.execute("""
        SELECT * FROM employees 
        WHERE department = %s AND is_active = TRUE
        ORDER BY name
    """, (team,))
    team_employees = cur.fetchall()
    
    cur.close()
    
    return render_template('bid_checklist.html',
                         team=team,
                         bid=bid,
                         checklist_items=checklist_items,
                         team_employees=team_employees,
                         user=current_user)

@app.route('/team/<team>/bids/<int:g_id>/checklist/create', methods=['POST'])
@login_required
def create_checklist_item(team, g_id):
    """Create a new checklist item for a bid"""
    try:
        task_name = request.form.get('task_name', '').strip()
        description = request.form.get('description', '').strip()
        assigned_to = request.form.get('assigned_to')
        priority = request.form.get('priority', 'medium')
        due_date = request.form.get('due_date')
        
        if not task_name:
            flash('Task name is required', 'error')
            return redirect(url_for('bid_checklist', team=team, g_id=g_id))
        
        cur = mysql.connection.cursor()
        # Explicit stage name for parallel tracking
        stage_name = team.strip().lower()
        cur.execute("""
            INSERT INTO bid_checklists (g_id, task_name, description, assigned_to, priority, due_date, progress_pct, stage, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (g_id, task_name, description, assigned_to if assigned_to else None, 
              priority, due_date if due_date else None, 0, stage_name, current_user.id))
        
        mysql.connection.commit()
        cur.close()
        
        log_write('create_checklist_item', f"Created task '{task_name}' for bid {g_id}")
        flash(f'Task "{task_name}" created successfully!', 'success')
        
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        flash(f'Error creating task: {str(e)}', 'error')
    
    return redirect(url_for('bid_checklist', team=team, g_id=g_id))

@app.route('/team/<team>/transfer_project', methods=['POST'])
@login_required
def transfer_project(team):
    """Transfer a project to another team"""
    try:
        g_id = request.form.get('g_id')
        to_team = request.form.get('to_team')
        transfer_reason = request.form.get('transfer_reason', '').strip()
        
        if not g_id or not to_team:
            flash('Project and destination team are required', 'error')
            return redirect(url_for('team_dashboard', team=team))
        
        cur = mysql.connection.cursor(DictCursor)
        
        # Get bid info
        cur.execute("SELECT * FROM go_bids WHERE g_id = %s", (g_id,))
        bid = cur.fetchone()
        
        if not bid:
            cur.close()
            flash('Bid not found', 'error')
            return redirect(url_for('team_dashboard', team=team))
        
        # Create transfer record
        cur.execute("""
            INSERT INTO project_transfers (g_id, from_team, to_team, transferred_by, transfer_reason)
            VALUES (%s, %s, %s, %s, %s)
        """, (g_id, team, to_team, current_user.id, transfer_reason))
        
        # Update bid state to next team
        cur.execute("UPDATE go_bids SET state = %s WHERE g_id = %s", (to_team, g_id))

        # When transferring, preserve current team's tasks and generate new tasks for receiving team
        # First, mark current team's tasks as completed and archive them
        cur.execute("""
            UPDATE bid_checklists 
            SET status = 'completed', progress_pct = 100, team_archive = %s, updated_at = CURRENT_TIMESTAMP
            WHERE g_id = %s AND created_by IN (
                SELECT id FROM users WHERE role = %s
            )
        """, (team, g_id, team))
        
        # Generate team-specific default checklist for the receiving team
        generate_team_checklist(cur, g_id, to_team)
        
        mysql.connection.commit()
        cur.close()
        
        log_write('project_transfer', 
                 f"Transferred bid '{bid['b_name']}' from {team} to {to_team}")
        flash(f'Project "{bid["b_name"]}" transferred to {to_team} team successfully!', 'success')
        
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        flash(f'Error transferring project: {str(e)}', 'error')
    
    return redirect(url_for('team_dashboard', team=team))

# --- API Endpoints for Team Dashboard ---
@app.route('/api/team/<team>/employees')
@login_required
def api_team_employees(team):
    """API endpoint to get team employees"""
    cur = mysql.connection.cursor(DictCursor)
    
    cur.execute("""
        SELECT e.*, u.email as team_lead_email
        FROM employees e
        LEFT JOIN users u ON e.team_lead_id = u.id
        WHERE e.department = %s AND e.is_active = TRUE
        ORDER BY e.name
    """, (team,))
    employees = cur.fetchall()
    
    cur.close()
    
    return jsonify({'employees': employees})

@app.route('/api/team/<team>/bids/<int:g_id>/tasks')
@login_required
def api_bid_tasks(team, g_id):
    """API endpoint to get tasks for a specific bid - only show tasks created by current team"""
    cur = mysql.connection.cursor(DictCursor)
    
    # Allow access if bid is in this team OR there are tasks for this team's stage
    cur.execute("SELECT state FROM go_bids WHERE g_id = %s", (g_id,))
    bid = cur.fetchone()
    cur.execute("SELECT 1 FROM bid_checklists WHERE g_id=%s AND LOWER(COALESCE(stage,''))=%s LIMIT 1", (g_id, team))
    has_team_tasks = cur.fetchone() is not None
    if not bid and not has_team_tasks:
        cur.close();
        return jsonify({'tasks': []})
    if bid and bid.get('state') != team and not has_team_tasks:
        cur.close();
        return jsonify({'tasks': []})
    
    # Get checklist items for this bid
    # Show tasks where the task's explicit stage matches this team (active), or archived by this team
    cur.execute("""
        SELECT bc.*, e.name as assigned_employee_name, e.email as employee_email, e.department
        FROM bid_checklists bc
        LEFT JOIN employees e ON bc.assigned_to = e.id
        WHERE bc.g_id = %s 
          AND (
            (LOWER(COALESCE(bc.stage,'')) = %s AND bc.team_archive IS NULL)
            OR bc.team_archive = %s
          )
        ORDER BY bc.priority DESC, bc.created_at ASC
    """, (g_id, team, team))
    tasks = cur.fetchall()

    # Emit per-stage map for this bid so master dashboard reflects progress
    try:
        def _pct_rows(rows):
            if not rows:
                return 0
            vals = []
            for r in rows:
                pct = r.get('progress_pct')
                if pct is None:
                    s = (r.get('status') or '').lower()
                    pct = 100 if s == 'completed' else 50 if s == 'in_progress' else 0
                try:
                    vals.append(max(0, min(100, int(pct))))
                except Exception:
                    vals.append(0)
            return int(round(sum(vals) / max(1, len(vals))))

        def _fetch_stage(stage_key):
            cur.execute("SELECT progress_pct, status FROM bid_checklists WHERE g_id=%s AND LOWER(COALESCE(stage,''))=%s", (g_id, stage_key))
            return cur.fetchall()

        spm = {
            'business': _pct_rows(_fetch_stage('business')),
            'design': _pct_rows(_fetch_stage('design')),
            'operations': _pct_rows(_fetch_stage('operations')),
            'engineer': _pct_rows(_fetch_stage('engineer')),
        }
        socketio.emit('master_update', {
            'summary': {
                'bid_id': g_id,
                'work_progress_pct': spm.get(team, 0),
                'project_status': 'ongoing',
                'work_status': f'{team.title()} tasks updated',
                'stage_progress_map': spm
            }
        })
    except Exception:
        pass

    cur.close()
    return jsonify({'tasks': tasks})

@app.route('/database-management/delete/<int:row_id>')
@login_required
def dbm_delete(row_id):
    if not current_user.is_admin:
        return "Access Denied", 403
    table = request.args.get('table')
    if not table:
        return redirect(url_for('database_management'))
    cur = mysql.connection.cursor(DictCursor)
    try:
        cur.execute(f"DESCRIBE `{table}`")
        cols = cur.fetchall()
        pk = next((c['Field'] for c in cols if c['Key'] == 'PRI'), cols[0]['Field'])
        
        # Handle foreign key constraints for specific tables
        if table == 'go_bids':
            # Delete related records first to avoid foreign key constraint errors
            cur.execute("DELETE FROM bid_checklists WHERE g_id=%s", (row_id,))
            cur.execute("DELETE FROM bid_stage_exclusions WHERE g_id=%s", (row_id,))
            cur.execute("DELETE FROM bid_custom_stages WHERE g_id=%s", (row_id,))
            
            # Delete from bid_assign (which may have win_lost_results, won_bids_result, work_progress_status cascading)
            cur.execute("SELECT a_id FROM bid_assign WHERE g_id=%s", (row_id,))
            assign_ids = [row['a_id'] for row in cur.fetchall()]
            
            for a_id in assign_ids:
                # Get w_id from win_lost_results
                cur.execute("SELECT w_id FROM win_lost_results WHERE a_id=%s", (a_id,))
                w_ids = [row['w_id'] for row in cur.fetchall()]
                
                for w_id in w_ids:
                    # Get won_id from won_bids_result
                    cur.execute("SELECT won_id FROM won_bids_result WHERE w_id=%s", (w_id,))
                    won_ids = [row['won_id'] for row in cur.fetchall()]
                    
                    # Delete work_progress_status
                    for won_id in won_ids:
                        cur.execute("DELETE FROM work_progress_status WHERE won_id=%s", (won_id,))
                    
                    # Delete won_bids_result
                    cur.execute("DELETE FROM won_bids_result WHERE w_id=%s", (w_id,))
                
                # Delete win_lost_results
                cur.execute("DELETE FROM win_lost_results WHERE a_id=%s", (a_id,))
            
            # Delete bid_assign
            cur.execute("DELETE FROM bid_assign WHERE g_id=%s", (row_id,))
        
        elif table == 'bid_incoming':
            # Find related go_bids and delete them (which will cascade)
            cur.execute("SELECT g_id FROM go_bids WHERE b_id=%s", (row_id,))
            g_ids = [row['g_id'] for row in cur.fetchall()]
            
            for g_id in g_ids:
                # Recursively delete go_bids using the same logic
                cur.execute("DELETE FROM bid_checklists WHERE g_id=%s", (g_id,))
                cur.execute("DELETE FROM bid_stage_exclusions WHERE g_id=%s", (g_id,))
                cur.execute("DELETE FROM bid_custom_stages WHERE g_id=%s", (g_id,))
                
                cur.execute("SELECT a_id FROM bid_assign WHERE g_id=%s", (g_id,))
                assign_ids = [row['a_id'] for row in cur.fetchall()]
                
                for a_id in assign_ids:
                    cur.execute("SELECT w_id FROM win_lost_results WHERE a_id=%s", (a_id,))
                    w_ids = [row['w_id'] for row in cur.fetchall()]
                    
                    for w_id in w_ids:
                        cur.execute("SELECT won_id FROM won_bids_result WHERE w_id=%s", (w_id,))
                        won_ids = [row['won_id'] for row in cur.fetchall()]
                        
                        for won_id in won_ids:
                            cur.execute("DELETE FROM work_progress_status WHERE won_id=%s", (won_id,))
                        
                        cur.execute("DELETE FROM won_bids_result WHERE w_id=%s", (w_id,))
                    
                    cur.execute("DELETE FROM win_lost_results WHERE a_id=%s", (a_id,))
                
                cur.execute("DELETE FROM bid_assign WHERE g_id=%s", (g_id,))
                cur.execute("DELETE FROM go_bids WHERE g_id=%s", (g_id,))
        
        elif table == 'bid_assign':
            # Get w_id from win_lost_results
            cur.execute("SELECT w_id FROM win_lost_results WHERE a_id=%s", (row_id,))
            w_ids = [row['w_id'] for row in cur.fetchall()]
            
            for w_id in w_ids:
                # Get won_id from won_bids_result
                cur.execute("SELECT won_id FROM won_bids_result WHERE w_id=%s", (w_id,))
                won_ids = [row['won_id'] for row in cur.fetchall()]
                
                # Delete work_progress_status
                for won_id in won_ids:
                    cur.execute("DELETE FROM work_progress_status WHERE won_id=%s", (won_id,))
                
                # Delete won_bids_result
                cur.execute("DELETE FROM won_bids_result WHERE w_id=%s", (w_id,))
            
            # Delete win_lost_results
            cur.execute("DELETE FROM win_lost_results WHERE a_id=%s", (row_id,))
        
        elif table == 'win_lost_results':
            # Get won_id from won_bids_result
            cur.execute("SELECT won_id FROM won_bids_result WHERE w_id=%s", (row_id,))
            won_ids = [row['won_id'] for row in cur.fetchall()]
            
            # Delete work_progress_status
            for won_id in won_ids:
                cur.execute("DELETE FROM work_progress_status WHERE won_id=%s", (won_id,))
            
            # Delete won_bids_result
            cur.execute("DELETE FROM won_bids_result WHERE w_id=%s", (row_id,))
        
        elif table == 'won_bids_result':
            # Delete work_progress_status
            cur.execute("DELETE FROM work_progress_status WHERE won_id=%s", (row_id,))
        
        # Now delete the main record
        cur.execute(f"DELETE FROM `{table}` WHERE `{pk}`=%s", (row_id,))
        mysql.connection.commit()
        flash(f'Record deleted successfully from {table}!', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Error deleting record: {str(e)}', 'error')
    finally:
        cur.close()
    return redirect(url_for('database_management', table=table))

@app.route('/database-management/drop')
@login_required
def dbm_drop():
    if not current_user.is_admin:
        return "Access Denied", 403
    table = request.args.get('table')
    if not table:
        return redirect(url_for('database_management'))
    cur = mysql.connection.cursor()
    try:
        cur.execute(f"DROP TABLE IF EXISTS `{table}`")
        mysql.connection.commit()
    finally:
        cur.close()
    return redirect(url_for('database_management'))

@app.route('/database-management/export')
@login_required
def dbm_export():
    if not current_user.is_admin:
        return "Access Denied", 403
    table = request.args.get('table')
    if not table:
        return redirect(url_for('database_management'))
    try:
        import pandas as pd
        import io
        cur = mysql.connection.cursor(DictCursor)
        cur.execute(f"SELECT * FROM `{table}`")
        rows = cur.fetchall()
        cur.close()
        df = pd.DataFrame(rows)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name=table[:31])
        output.seek(0)
        filename = f"{table}.xlsx"
        return send_file(output, as_attachment=True, download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        flash(f'Export failed: {str(e)}', 'error')
        return redirect(url_for('database_management', table=table))

@app.route('/database-management/import', methods=['POST'])
@login_required
def dbm_import():
    if not current_user.is_admin:
        return "Access Denied", 403
    table = request.form.get('table')
    file = request.files.get('excel_file')
    if not table or not file or file.filename == '':
        flash('Please choose a table and an Excel file.', 'error')
        return redirect(url_for('database_management', table=table or ''))
    try:
        import pandas as pd
        import io
        df = pd.read_excel(io.BytesIO(file.read()))
        cur = mysql.connection.cursor(DictCursor)
        # Fetch columns and ignore primary key
        cur.execute(f"DESCRIBE `{table}`")
        cols = cur.fetchall()
        non_pk_cols = [c['Field'] for c in cols if c['Key'] != 'PRI']
        # Filter dataframe to only known columns
        df = df[[c for c in df.columns if c in non_pk_cols]]
        if df.empty:
            flash('No matching columns found in Excel file for this table.', 'error')
            cur.close()
            return redirect(url_for('database_management', table=table))
        placeholders = ','.join(['%s'] * len(df.columns))
        fields_sql = ','.join([f"`{c}`" for c in df.columns])
        sql = f"INSERT INTO `{table}` ({fields_sql}) VALUES ({placeholders})"
        for _, row in df.iterrows():
            cur.execute(sql, [None if pd.isna(v) else v for v in row.tolist()])
        mysql.connection.commit()
        cur.close()
        flash(f'Successfully imported {len(df)} rows into `{table}`.', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Import failed: {str(e)}', 'error')
    return redirect(url_for('database_management', table=table))

# --- API & Real-time Logic ---
@app.route('/api/update_stage/<int:bid_id>', methods=['POST'])
@login_required
def update_stage(bid_id):
    try:
        data = request.get_json()
        new_stage = (data.get('stage') or '').lower()
        
        # Validate stage
        allowed = {'analyzer', 'business', 'design', 'operations', 'engineer', 'handover'}
        if new_stage not in allowed:
            return jsonify({'error': 'invalid stage'}), 400
        
        cur = mysql.connection.cursor(DictCursor)
        
        # Get bid information from go_bids
        cur.execute("SELECT * FROM go_bids WHERE g_id=%s", (bid_id,))
        bid = cur.fetchone()
        
        if not bid:
            cur.close()
            return jsonify({'error': 'Bid not found or access denied'}), 404

        # Get old stage for logging
        old_stage = (bid.get('state') or 'analyzer').lower()
        
        # Update go_bids state
        cur.execute("UPDATE go_bids SET state=%s WHERE g_id=%s", (new_stage, bid_id))
        
        # Upsert assignment so the next team still sees it
        cur.execute("SELECT a_id FROM bid_assign WHERE g_id=%s", (bid_id,))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE bid_assign SET depart=%s, state=%s, status='pending' WHERE g_id=%s",
                        (new_stage, new_stage, bid_id))
        else:
            cur.execute("""
                INSERT INTO bid_assign (g_id, b_name, in_date, due_date, state, scope, type, company, depart,
                                       person_name, assignee_email, status, value, revenue)
                SELECT g_id, b_name, in_date, due_date, state, scope, type, company, %s, '', '', 'pending',
                       COALESCE(scoring, 0), COALESCE(revenue, 0)
                FROM go_bids WHERE g_id=%s
            """, (new_stage, bid_id))
        
        # Derive dynamic summary line
        from_txt = LABELS.get(old_stage, '')
        to_txt = LABELS.get(new_stage, '')
        summary_line = f"Updated by {from_txt} to {to_txt}"
        
        # Commit the transaction
        mysql.connection.commit()
        
        # Log the stage change
        log_write('stage_change', f"{bid.get('b_name')} | {old_stage} → {new_stage}")
        
        # Calculate dynamic progress and status texts for new stage
        pct = pct_for(new_stage)
        proj_status = 'completed' if new_stage == 'handover' else 'ongoing'
        
        # Recompute cards and analyzer stats after update
        # Summary cards across the three companies only
        cur2 = mysql.connection.cursor(DictCursor)
        cur2.execute("SELECT id FROM companies WHERE name IN ('Ikio','Metco','Sunsprint')")
        target_company_ids = [row['id'] for row in cur2.fetchall()]
        if not target_company_ids:
            target_company_ids = [-1]
        in_clause = ','.join(['%s'] * len(target_company_ids))

        # Compute totals from go_bids instead of bids
        cur2.execute("SELECT name FROM companies WHERE name IN ('Ikio','Metco','Sunsprint')")
        target_company_names = [row['name'] for row in cur2.fetchall()] or ['__none__']
        in_clause_names = ','.join(['%s'] * len(target_company_names))
        cur2.execute(f"SELECT COUNT(*) AS total_bids FROM go_bids WHERE company IN ({in_clause_names})", target_company_names)
        total_bids = cur2.fetchone()['total_bids']
        cur2.execute(f"SELECT COUNT(*) AS live_bids FROM go_bids WHERE COALESCE(state,'analyzer') IN ('business','design','operations','engineer') AND company IN ({in_clause_names})", target_company_names)
        live_bids = cur2.fetchone()['live_bids']
        cur2.execute(f"SELECT COUNT(*) AS bids_won FROM go_bids WHERE decision='WON' AND company IN ({in_clause_names})", target_company_names)
        bids_won = cur2.fetchone()['bids_won']
        # Total projects across target companies (fallback to all projects if none linked)
        cur2.execute(f"SELECT COUNT(*) AS projects_linked FROM projects WHERE company_id IN ({in_clause})", target_company_ids)
        projects_linked = cur2.fetchone()['projects_linked']
        if projects_linked > 0:
            cur2.execute(f"SELECT COUNT(*) AS projects_total FROM projects WHERE company_id IN ({in_clause})", target_company_ids)
            projects_total = cur2.fetchone()['projects_total']
        else:
            cur2.execute("SELECT COUNT(*) AS projects_total FROM projects")
            projects_total = cur2.fetchone()['projects_total']

        # Analyzer stats from bid_incoming table
        cur2.execute("SELECT COUNT(*) AS total_bids FROM bid_incoming")
        total_bids_analyzer = cur2.fetchone()['total_bids']
        
        cur2.execute("SELECT COUNT(*) AS bids_go FROM bid_incoming WHERE decision = 'GO'")
        bids_go_analyzer = cur2.fetchone()['bids_go']
        
        cur2.execute("SELECT COUNT(*) AS bids_no_go FROM bid_incoming WHERE decision = 'NO-GO'")
        bids_no_go_analyzer = cur2.fetchone()['bids_no_go']
        
        cur2.execute("SELECT COUNT(*) AS bids_submitted FROM bid_incoming WHERE state IN ('submitted', 'under_review')")
        bids_submitted_analyzer = cur2.fetchone()['bids_submitted']
        
        cur2.execute("SELECT COUNT(*) AS bids_won FROM bid_incoming WHERE decision = 'WON'")
        bids_won_analyzer = cur2.fetchone()['bids_won']
        
        cur2.execute("SELECT COUNT(*) AS bids_lost FROM bid_incoming WHERE decision = 'LOST'")
        bids_lost_analyzer = cur2.fetchone()['bids_lost']

        bid_stats = {
            'total_bids': total_bids_analyzer,
            'bids_go': bids_go_analyzer,
            'bids_no_go': bids_no_go_analyzer,
            'bids_submitted': bids_submitted_analyzer,
            'bids_won': bids_won_analyzer,
            'bids_lost': bids_lost_analyzer
        }

        socketio.emit('master_update', {
            'bid': {
                'id': bid_id,
                'name': bid.get('b_name'),
                'current_stage': new_stage,
                'user_email': getattr(current_user, 'email', '')
            },
            'summary': {
                'work_progress_pct': pct,
                'project_status': proj_status,
                'work_status': summary_line
            },
            'cards': {
                'total_bids': projects_total,
                'live_bids': 0,
                'bids_won': 0,
                'projects_completed': 0
            },
            'bid_stats': bid_stats
        })
        
        cur.close()
        cur2.close()
        return jsonify({'success': f'Bid {bid_id} updated to {new_stage}'})
    
    except Exception as e:
        mysql.connection.rollback()
        if 'cur' in locals():
            cur.close()
        if 'cur2' in locals():
            cur2.close()
        return jsonify({'error': f'Error updating stage: {str(e)}'}), 500

# --- Main execution ---
def _ensure_tables_exist():
    """Create tables if they don't exist"""
    try:
        cur = mysql.connection.cursor()
        
        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(100) NOT NULL,
                is_admin BOOLEAN DEFAULT FALSE,
                role VARCHAR(50) DEFAULT 'member'
            )
        """)
        
        # Create bids table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bids (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                current_stage VARCHAR(50) DEFAULT 'analyzer',
                user_id INT,
                company_id INT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Create bid_incoming table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bid_incoming (
                b_id INT AUTO_INCREMENT PRIMARY KEY,
                b_name VARCHAR(100),
                in_date DATE DEFAULT CURRENT_TIMESTAMP,
                due_date DATE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                state VARCHAR(100),
                scope TEXT,
                type VARCHAR(100),
                scoring INT,
                comp_name VARCHAR(100),
                decision VARCHAR(100),
                summary TEXT
            )
        """)
        
        # Create go_bids table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS go_bids (
                g_id INT AUTO_INCREMENT PRIMARY KEY,
                b_id INT,
                b_name VARCHAR(100),
                in_date DATE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                due_date DATE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                state VARCHAR(100),
                scope VARCHAR(100),
                type VARCHAR(100),
                scoring INT,
                company TEXT,
                decision TEXT,
                summary TEXT,
                revenue DECIMAL(15,2) DEFAULT 0.00
            )
        """)
        
        # Create bid_assign table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bid_assign (
                a_id INT AUTO_INCREMENT PRIMARY KEY,
                g_id INT,
                b_name VARCHAR(100),
                in_date DATE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                due_date DATE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                state VARCHAR(100),
                scope VARCHAR(100),
                type VARCHAR(100),
                company TEXT,
                depart TEXT,
                person_name TEXT,
                assignee_email VARCHAR(100),
                status TEXT,
                value INT,
                revenue DECIMAL(15,2) DEFAULT 0.00
            )
        """)
        
        # Create win_lost_results table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS win_lost_results (
                w_id INT AUTO_INCREMENT PRIMARY KEY,
                a_id INT,
                b_name TEXT,
                in_date INT,
                due_date INT,
                state TEXT,
                scope TEXT,
                value INT,
                company TEXT,
                department TEXT,
                person_name TEXT,
                status TEXT,
                result TEXT
            )
        """)
        
        # Create won_bids_result table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS won_bids_result (
                won_id INT AUTO_INCREMENT PRIMARY KEY,
                w_id INT,
                closure_status TEXT,
                work_progress_status TEXT
            )
        """)
        
        # Create assigned_bids table using the exact schema requested
        cur.execute("""
            CREATE TABLE IF NOT EXISTS assigned_bids (
                id INT AUTO_INCREMENT PRIMARY KEY,
                g_id INT,
                b_name VARCHAR(100),
                company VARCHAR(100),
                revenue DECIMAL(15,2) DEFAULT 0.00,
                assigned_to VARCHAR(100),
                assigned_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create work_progress_status table (extended schema used across the app)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS work_progress_status (
                p_id INT AUTO_INCREMENT PRIMARY KEY,
                won_id INT,
                company TEXT,
                b_name TEXT,
                dept_bde TEXT,
                dept_m_d TEXT,
                dept_op TEXT,
                dept_site TEXT,
                pr_completion_status TEXT
            )
        """)

    
        
        # Create logs table for tracking user actions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                action VARCHAR(255) NOT NULL,
                user_id INT,
                timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Create companies table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create projects table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                company_id INT NOT NULL,
                start_date DATETIME,
                due_date DATETIME NOT NULL,
                revenue FLOAT DEFAULT 0.0,
                status VARCHAR(50) DEFAULT 'active',
                progress INT DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies(id)
            )
        """)
        
        # Create tasks table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                project_id INT NOT NULL,
                assigned_user_id INT,
                due_date DATETIME NOT NULL,
                status VARCHAR(50) DEFAULT 'pending',
                priority VARCHAR(20) DEFAULT 'medium',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id),
                FOREIGN KEY (assigned_user_id) REFERENCES users(id)
            )
        """)
        
        # Create bid timeline table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bid_timeline (
                id INT AUTO_INCREMENT PRIMARY KEY,
                bid_id INT,
                event VARCHAR(200) NOT NULL,
                details TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create employees table for team-specific employee management
        cur.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                department VARCHAR(50) NOT NULL,
                team_lead_id INT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_lead_id) REFERENCES users(id)
            )
        """)
        
        # Create bid_checklists table for task management per bid
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bid_checklists (
                id INT AUTO_INCREMENT PRIMARY KEY,
                g_id INT NOT NULL,
                task_name VARCHAR(200) NOT NULL,
                description TEXT,
                assigned_to INT,
                status VARCHAR(50) DEFAULT 'pending',
                progress_pct INT DEFAULT NULL,
                priority VARCHAR(20) DEFAULT 'medium',
                due_date DATETIME,
                created_by INT,
                team_archive VARCHAR(50),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (g_id) REFERENCES go_bids(g_id),
                FOREIGN KEY (assigned_to) REFERENCES employees(id),
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        """)

        # Ensure progress_pct column exists even on older databases
        cur.execute("SELECT COUNT(*) AS cnt FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='bid_checklists' AND COLUMN_NAME='progress_pct'")
        row = cur.fetchone()
        if not row or int(row.get('cnt', 0)) == 0:
            try:
                cur.execute("ALTER TABLE bid_checklists ADD COLUMN progress_pct INT NULL AFTER status")
            except Exception:
                pass
        
        # Create project_transfers table for tracking project handoffs between teams
        cur.execute("""
            CREATE TABLE IF NOT EXISTS project_transfers (
                id INT AUTO_INCREMENT PRIMARY KEY,
                g_id INT NOT NULL,
                from_team VARCHAR(50) NOT NULL,
                to_team VARCHAR(50) NOT NULL,
                transferred_by INT,
                transfer_reason TEXT,
                status VARCHAR(50) DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (g_id) REFERENCES go_bids(g_id),
                FOREIGN KEY (transferred_by) REFERENCES users(id)
            )
        """)

        mysql.connection.commit()
        cur.close()
        print("Database tables created/verified successfully")
    except Exception as e:
        print(f"Error creating tables: {e}")
        try:
            mysql.connection.rollback()
        except Exception:
            pass
        if 'cur' in locals():
            try:
                cur.close()
            except Exception:
                pass

if __name__ == '__main__':
    from datetime import datetime, timedelta
    
    with app.app_context():
        _ensure_tables_exist()
        
        cur = mysql.connection.cursor(DictCursor)
        
        # Always ensure companies exist
        cur.execute("SELECT COUNT(*) as count FROM companies")
        if cur.fetchone()['count'] == 0:
            cur.execute("""
                INSERT INTO companies (name, description) VALUES 
                ('Ikio', 'Renewable Energy Solutions'),
                ('Metco', 'Industrial Energy Management'),
                ('Sunsprint', 'Solar Power Systems')
            """)
            mysql.connection.commit()
            print("Companies created successfully")
        
        # Check if users exist
        cur.execute("SELECT COUNT(*) as count FROM users")
        if cur.fetchone()['count'] == 0:
            # Create admin user
            cur.execute("""
                INSERT INTO users (email, password, is_admin, role) VALUES 
                ('admin@example.com', 'admin', 1, 'admin')
            """)
            admin_user_id = cur.lastrowid
            
            # Create other users
            cur.execute("""
                INSERT INTO users (email, password, is_admin, role) VALUES 
                ('bd@example.com', 'user', 0, 'business dev'),
                ('designer@example.com', 'designer', 0, 'design'),
                ('ops@example.com', 'ops', 0, 'operations'),
                ('sitemgr@example.com', 'site', 0, 'site manager')
            """)
            # Fetch exact user IDs by email to avoid relying on lastrowid math
            cur.execute("SELECT id FROM users WHERE email=%s", ('bd@example.com',))
            user_bdm_id = cur.fetchone()['id']
            cur.execute("SELECT id FROM users WHERE email=%s", ('designer@example.com',))
            user_design_id = cur.fetchone()['id']
            cur.execute("SELECT id FROM users WHERE email=%s", ('ops@example.com',))
            user_ops_id = cur.fetchone()['id']
            cur.execute("SELECT id FROM users WHERE email=%s", ('sitemgr@example.com',))
            user_site_id = cur.fetchone()['id']
            
            # Get company ids for linking bids
            cur.execute("SELECT id FROM companies WHERE name='Ikio'")
            ikio_id = cur.fetchone()['id']
            cur.execute("SELECT id FROM companies WHERE name='Metco'")
            metco_id = cur.fetchone()['id']
            cur.execute("SELECT id FROM companies WHERE name='Sunsprint'")
            sunsprint_id = cur.fetchone()['id']

            # Create sample bids linked to companies
            cur.execute("""
            INSERT INTO bids (name, current_stage, user_id, company_id) VALUES 
            ('Project Alpha', 'business', %s, %s),
            ('Project Beta', 'design', %s, %s),
            ('Project Gamma', 'operations', %s, %s)
        """, (user_bdm_id, ikio_id, user_design_id, metco_id, user_ops_id, sunsprint_id))
            # Create sample bid_incoming data
            cur.execute("""
                INSERT INTO bid_incoming (b_name, due_date, state, scope, type, scoring, comp_name, decision, summary) VALUES 
                ('Solar Energy Project', %s, 'submitted', 'Installation of 500kW solar panels for commercial building', 'Renewable Energy', 85, 'Ikio', 'GO', 'High potential project with good ROI'),
                ('Wind Farm Development', %s, 'under_review', 'Development of 2MW wind farm in rural area', 'Wind Energy', 72, 'Metco', 'NO-GO', 'Land acquisition issues identified'),
                ('Energy Efficiency Audit', %s, 'pending', 'Comprehensive energy audit for manufacturing facility', 'Energy Management', 90, 'Sunsprint', 'WON', 'Excellent technical proposal with competitive pricing'),
                ('Battery Storage System', %s, 'submitted', 'Installation of 1MWh battery storage system', 'Energy Storage', 78, 'Ikio', 'LOST', 'Lost to competitor with lower bid'),
                ('Smart Grid Implementation', %s, 'completed', 'Implementation of smart grid technology for city', 'Smart Grid', 95, 'Metco', 'WON', 'Successfully completed project ahead of schedule')
            """, (
                datetime.now() + timedelta(days=30),
                datetime.now() + timedelta(days=45),
                datetime.now() + timedelta(days=15),
                datetime.now() + timedelta(days=60),
                datetime.now() + timedelta(days=90)
            ))
            
            # One sample bid ready for site manager handover
            cur.execute("UPDATE bids SET current_stage='site_manager' WHERE name='Project Beta'")
            
            mysql.connection.commit()
            
            # Company ids already loaded above
            
            # Create sample projects
            cur.execute("""
                INSERT INTO projects (name, company_id, start_date, due_date, revenue, status, progress) VALUES 
                ('Solar Farm Installation', %s, %s, %s, 50000, 'active', 45),
                ('Wind Energy Project', %s, %s, %s, 75000, 'active', 70),
                ('Energy Efficiency Audit', %s, %s, %s, 25000, 'active', 30),
                ('Industrial Solar Setup', %s, %s, %s, 100000, 'active', 15),
                ('Residential Solar Panel', %s, %s, %s, 30000, 'active', 80),
                ('Commercial Solar System', %s, %s, %s, 60000, 'active', 25)
            """, (
                ikio_id, datetime.now(), datetime.now() + timedelta(days=90),
                ikio_id, datetime.now() - timedelta(days=30), datetime.now() + timedelta(days=60),
                metco_id, datetime.now() - timedelta(days=15), datetime.now() + timedelta(days=45),
                metco_id, datetime.now(), datetime.now() + timedelta(days=120),
                sunsprint_id, datetime.now() - timedelta(days=10), datetime.now() + timedelta(days=30),
                sunsprint_id, datetime.now(), datetime.now() + timedelta(days=75)
            ))
            
            # Get project IDs for tasks
            cur.execute("SELECT id FROM projects ORDER BY id")
            project_ids = [row['id'] for row in cur.fetchall()]
            
            # Create sample tasks
            cur.execute("""
                INSERT INTO tasks (name, project_id, assigned_user_id, due_date, status, priority) VALUES 
                ('Site Survey', %s, %s, %s, 'in_progress', 'high'),
                ('Equipment Procurement', %s, %s, %s, 'pending', 'medium'),
                ('Installation Planning', %s, %s, %s, 'completed', 'high'),
                ('Energy Assessment', %s, %s, %s, 'in_progress', 'urgent'),
                ('Client Consultation', %s, %s, %s, 'pending', 'high'),
                ('System Testing', %s, %s, %s, 'in_progress', 'medium'),
                ('Documentation', %s, %s, %s, 'pending', 'low')
            """, (
                project_ids[0], user_bdm_id, datetime.now() + timedelta(days=5),
                project_ids[0], user_design_id, datetime.now() + timedelta(days=15),
                project_ids[1], admin_user_id, datetime.now() + timedelta(days=10),
                project_ids[2], user_ops_id, datetime.now() + timedelta(days=7),
                project_ids[3], user_site_id, datetime.now() + timedelta(days=3),
                project_ids[4], admin_user_id, datetime.now() + timedelta(days=2),
                project_ids[5], user_bdm_id, datetime.now() + timedelta(days=20)
            ))
            
            mysql.connection.commit()
            
        # Seed at least one log if none exist
        cur.execute("SELECT COUNT(*) as count FROM logs")
        if cur.fetchone()['count'] == 0:
            cur.execute("INSERT INTO logs (action) VALUES ('System initialized and sample data seeded.')")
            mysql.connection.commit()
        
        cur.close()
    
    socketio.run(app, debug=True, port=5001)
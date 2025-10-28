# ====================================================================
# INTEGRATION SNIPPET FOR app_v2.py
# Add these lines to integrate the RFP Analyzer
# ====================================================================

# ========== 1. ADD THIS IMPORT AT THE TOP (around line 11) ==========
from rfp_analyzer_routes import rfp_bp

# ========== 2. REGISTER BLUEPRINT (add after app initialization, around line 15) ==========
app.register_blueprint(rfp_bp)

# ========== 3. UPDATE BID ANALYZER ROUTE (replace existing @app.route('/bid-analyzer')) ==========
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

# ========== 4. SET MAX FILE UPLOAD SIZE (add after app.config['SECRET_KEY']) ==========
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file upload

# ====================================================================
# END OF INTEGRATION SNIPPET
# ====================================================================

# ====================================================================
# ALTERNATIVE: QUICK MINIMAL INTEGRATION
# If you want minimal changes, just add these 2 lines:
# ====================================================================

# At top of file (imports):
# from rfp_analyzer_routes import rfp_bp

# After app initialization:
# app.register_blueprint(rfp_bp)

# That's it! The /rfp-analyzer/ route will be automatically available
# ====================================================================


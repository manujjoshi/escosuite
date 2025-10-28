# Team Sub-Dashboards Implementation

## Overview
This implementation adds role-based team sub-dashboards to the ESCO Intelligent Suite, allowing different teams to manage their specific stage of the bid process with real-time updates and activity logging.

## Features Implemented

### 1. Role-Based Login Redirects
- **Admin users** → `/master-dashboard`
- **Business Dev** → `/dashboard/business`
- **Design** → `/dashboard/design`
- **Operations** → `/dashboard/operations`
- **Site Manager** → `/dashboard/engineer`

### 2. Team Dashboards
Each team dashboard (`/dashboard/<team>`) includes:
- **Team-specific bid filtering** based on `go_bids.state`
- **Statistics cards** showing total bids, completed bids, and in-progress bids
- **Interactive bid table** with progress bars and status indicators
- **Stage advancement** functionality with forward/backward movement
- **Real-time updates** via Socket.IO
- **Live activity log** showing recent actions

### 3. Stage Management
- **Role ↔ Stage Mapping**:
  - Business Dev → `business`
  - Design → `design`
  - Operations → `operations`
  - Site Manager → `engineer`
  - Handover → Admin only

### 4. Real-Time Updates
- **Socket.IO integration** for live updates
- **Master dashboard** receives updates when teams advance stages
- **Activity logging** with user, action, and timestamp
- **Timeline tracker** updates with blinking dots and progress lines

### 5. Database Schema
Added/Updated tables:
- **`logs`** - Tracks user actions and updates
- **`go_bids`** - Main bids table with state management
- **`bid_assign`** - Bid assignments
- **`win_lost_results`** - Win/loss tracking
- **`won_bids_result`** - Won bid details
- **`work_progress_status`** - Work progress tracking

## File Structure

### New Files
- `templates/team_dashboard.html` - Team dashboard template
- `test_team_dashboards.py` - Test suite
- `TEAM_DASHBOARDS_IMPLEMENTATION.md` - This documentation

### Modified Files
- `app_v2.py` - Added team dashboard routes and enhanced logging
- `templates/master_dashboard.html` - Added live activity log section
- `models.py` - Already had role field (no changes needed)

## API Endpoints

### Authentication
- `POST /login` - Enhanced with role-based redirects

### Team Dashboards
- `GET /dashboard/<team>` - Team-specific dashboard
  - Teams: `business`, `design`, `operations`, `engineer`

### Stage Management
- `POST /api/update_stage/<int:bid_id>` - Update bid stage
  - Requires authentication
  - Logs action to database
  - Emits Socket.IO update
  - Returns JSON response

## Usage

### 1. Start the Application
```bash
python app_v2.py
```

### 2. Create Users
Use the admin panel at `/admin/users` to create users with appropriate roles:
- **Admin**: `is_admin=True`, `role='admin'`
- **Business Dev**: `role='business dev'`
- **Design**: `role='design'`
- **Operations**: `role='operations'`
- **Site Manager**: `role='site manager'`

### 3. Test the Implementation
```bash
python test_team_dashboards.py
```

## Real-Time Features

### Socket.IO Events
- **`master_update`** - Emitted when a bid stage is updated
  - Contains: bid info, log data, updated statistics
  - Received by: master dashboard and team dashboards

### Activity Logging
- **Automatic logging** of all stage updates
- **Real-time display** on master dashboard
- **User attribution** with email and role
- **Timestamp** for each action

## Security Considerations

### Access Control
- **Role-based access** to team dashboards
- **Admin-only** access to master dashboard
- **Authentication required** for all dashboard routes

### Data Validation
- **Stage validation** before updates
- **User permission checks** for stage changes
- **SQL injection protection** via parameterized queries

## Future Enhancements

### Potential Improvements
1. **Bid assignment** to specific team members
2. **Email notifications** for stage changes
3. **File uploads** for bid documents
4. **Advanced reporting** and analytics
5. **Mobile-responsive** design improvements
6. **Bulk operations** for multiple bids
7. **Custom workflows** per company
8. **Integration** with external systems

### Database Optimizations
1. **Indexing** on frequently queried columns
2. **Caching** for dashboard statistics
3. **Archiving** old logs and completed bids
4. **Backup strategies** for production data

## Troubleshooting

### Common Issues
1. **Socket.IO not connecting** - Check if server is running
2. **Database errors** - Verify MySQL connection and table structure
3. **Permission denied** - Check user roles and authentication
4. **Real-time updates not working** - Check browser console for errors

### Debug Mode
Enable Flask debug mode for detailed error messages:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## Conclusion

The team sub-dashboards implementation provides a comprehensive solution for role-based bid management with real-time updates and activity tracking. The system is designed to be scalable, secure, and user-friendly while maintaining the existing functionality of the ESCO Intelligent Suite.

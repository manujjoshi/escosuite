# Template Error Fix - get_next_stage Function

## Problem
The team dashboard was throwing a `jinja2.exceptions.UndefinedError: 'get_next_stage' is undefined` error when trying to render the template.

## Root Cause
The `team_dashboard.html` template was trying to use a `get_next_stage(current_stage)` function that wasn't defined or passed to the template context.

## Solution Implemented

### 1. Added get_next_stage Function
**File:** `app_v2.py` (team_dashboard function)

```python
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
```

### 2. Passed Function to Template Context
**File:** `app_v2.py` (team_dashboard function)

```python
return render_template('team_dashboard.html', 
                     bids=bids, 
                     team=team,
                     team_display_name=team_display_name,
                     current_stage=current_stage,
                     total_bids=total_bids,
                     completed_bids=completed_bids,
                     user=current_user,
                     get_next_stage=get_next_stage)  # Added this
```

### 3. Enhanced Template Logic
**File:** `templates/team_dashboard.html`

- Added conditional logic to handle cases where there's no next stage
- Shows "Complete" button instead of "Advance" for final stages
- Improved JavaScript error handling

```html
{% set next_stage = get_next_stage(current_stage) %}
{% if next_stage %}
<button onclick="advanceStage({{ bid.id }}, '{{ next_stage }}')">
    <i class="fas fa-arrow-right mr-1"></i>Advance
</button>
{% else %}
<span class="px-3 py-1.5 bg-gray-400 text-white rounded text-sm cursor-not-allowed">
    <i class="fas fa-check mr-1"></i>Complete
</span>
{% endif %}
```

### 4. Improved JavaScript Handling
**File:** `templates/team_dashboard.html`

```javascript
function advanceStage(bidId, nextStage) {
    if (!nextStage || nextStage === 'null' || nextStage === 'undefined') {
        alert('No next stage available - this is the final stage');
        return;
    }
    // ... rest of function
}
```

## Stage Flow Logic

The stage progression follows this flow:
- **analyzer** → **business**
- **business** → **design** 
- **design** → **operations**
- **operations** → **engineer**
- **engineer** → **handover** (final stage)

## Benefits

1. **No More Template Errors** - `get_next_stage` function is now properly defined
2. **Better UX** - Shows appropriate buttons for each stage
3. **Clear Stage Progression** - Users understand the workflow
4. **Error Prevention** - Handles edge cases gracefully

## Testing

1. **Navigate to team dashboards:**
   - `/dashboard/business`
   - `/dashboard/design`
   - `/dashboard/operations`
   - `/dashboard/engineer`

2. **Verify functionality:**
   - Advance buttons work for intermediate stages
   - Complete button shows for final stage
   - No template errors occur

## Status
✅ **FIXED** - The template error has been resolved and team dashboards should now load without issues.

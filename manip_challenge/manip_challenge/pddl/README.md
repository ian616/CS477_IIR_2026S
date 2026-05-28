# PDDL TAMP server

This folder contains an experimental PDDL-based task-and-motion planning layer.
It keeps symbolic planning separate from physical execution:

1. Natural language is parsed into target goals with Gemini or a rule-based fallback.
2. Top-view YOLO/RGB-D perception is converted into boolean predicates.
3. `domain.pddl` and `problem.pddl` are generated.
4. An external planner is used if `PDDL_PLANNER_CMD` is set; otherwise a small fallback planner chooses one action.
5. Only the first physical action is executed, then the scene is perceived again and replanned.

## Run

Terminal 1:

```bash
ros2 launch manip_challenge ur5_setup.launch.py
```

Terminal 2:

```bash
cd /home/lhs/CS477_IIR_2026S/manip_challenge/manip_challenge/pddl
python3 server.py
```

Terminal 3:

```bash
cd /home/lhs/CS477_IIR_2026S/manip_challenge/manip_challenge/pddl
python3 client.py "Move the banana and the meat can to the left storage. Move the strawberry and the hammer in the right storage. Move the coke can on the shelf."
```

The client publishes the command once as `std_msgs/String` on `/task_commands`.
You can publish without the client too:

```bash
ros2 topic pub --once /task_commands std_msgs/msg/String "{data: 'Move the banana to the left storage.'}"
```

The legacy service path is still available:

```bash
python3 client.py --service "Move the banana to the left storage."
```

Dry-run planning:

```bash
python3 server.py --dry-run
```

To use a different command topic:

```bash
python3 server.py --command-topic /my_task_commands
python3 client.py --command-topic /my_task_commands "Move the banana to the left storage."
```

Debug view:

```bash
python3 server.py --debug
python3 server.py --debug --debug-window
python3 server.py --debug --debug-window --debug-wait
```

`--debug` saves the latest perception image plus predicate judgement panel to:

```text
pddl/debug/latest_debug.png
```

`--debug-window` also opens an OpenCV window named `PDDL TAMP Debug` when a GUI
display is usable.

Combined PDDL log stream:

```bash
ros2 topic echo /pddl_tamp_log
```

This single `std_msgs/String` topic publishes plain-text log events for parsed
goals, object judgements, predicates, planner output, selected actions, action
results, replanning, and command completion. To change the topic name:

```bash
python3 server.py --log-topic /my_pddl_log
```

## Gemini

Edit `.env`:

```text
GEMINI_API_KEY=...
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash
```

If no key is configured, the rule-based parser still handles the standard move commands.

## External planner

Set `PDDL_PLANNER_CMD` in `.env`, using `{domain}` and `{problem}` placeholders.
If it is empty or fails, `planner.py` uses the fallback planner.

## Action handlers

All symbolic-to-physical mappings live in `actions.py`:

- `move-target-to-goal`
- `move-obstacle-to-buffer`

These handlers reuse existing `custom.motion` and `custom.grasping` behavior.
Obstacle movement requires the perception service to provide a usable grasp pose
for that obstacle label. BBox-only unknown obstacles are logged but not executed.

## Where to edit

- Change "what is true in the scene" in `predicate_builder.py`.
  This is where YOLO/RGB-D geometry becomes `visible`, `graspable`, `blocks`,
  `near`, `clear`, and `safe`.
- Change "what robot motion runs for a symbolic action" in `actions.py`.
  This is where `(move-target-to-goal banana table left_storage)` becomes a
  call into `custom.motion.motion.execute_pick_place_sequence()`.
- Change target grasp point selection in `server.py::select_grasp_target()` or
  `ros_helpers.py::choose_grasp_target_from_points()`.
- Change final grasp pose conversion in `server.py::make_grasp_pose()`.
- Change actual pick/place trajectories in `custom/motion/motion.py`.
- Change object-specific grasp transform and gripper close behavior in
  `custom/grasping/grasping_item.py`.

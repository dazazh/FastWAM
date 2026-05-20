# VLM CoT - Results

Run root: `evaluate_results/robotwin/vlm_matrix/20260519_115312`

Setup: FastWAM release ckpt `robotwin_uncond_3cam_384.pt`, `task_config=demo_randomized`, `instruction_type=unseen`. VLM = Qwen3-VL-Plus via DashScope. Open-loop (M2) calls VLM once per episode with the first head-frame; closed-loop (M3) calls VLM every K replan boundaries and either picks a subtask from a per-task menu scaffold (menu mode) or emits a free-form short instruction (free_form mode).

Conditions in this run: baseline, vlm, vlm_cl_menu, vlm_cl_free

## Success Rate

| Task | Baseline (M1, raw) | Open-loop VLM (M2) | Closed-loop menu (M3-menu) | Closed-loop free (M3-free) |
| --- | --- | --- | --- | --- |
| hanging_mug | 20.0% | 30.0% | MISSING | MISSING |
| open_microwave | 30.0% | 60.0% | 40.0% | 20.0% |
| place_can_basket | 30.0% | 60.0% | MISSING | MISSING |
| turn_switch | 40.0% | 40.0% | 70.0% | 70.0% |
| **mean** | **30.0%** | **47.5%** | **55.0%** | **45.0%** |

## Per-Task Details

### hanging_mug

**baseline** (Baseline (M1, raw)):
- success rate: **20.0%**  (2 success / 8 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/hanging_mug_baseline/hanging_mug/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/hanging_mug_baseline/hanging_mug/episode0_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/hanging_mug_baseline/hanging_mug/episode1_randomized-true_success-false.mp4`

**vlm** (Open-loop VLM (M2)):
- success rate: **30.0%**  (3 success / 7 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/hanging_mug_vlm/hanging_mug/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/hanging_mug_vlm/hanging_mug/episode0_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/hanging_mug_vlm/hanging_mug/episode1_randomized-true_success-false.mp4`
- VLM augmentations (episode_seed -> augmented_instruction):
  - `hanging_mug|seed=1` (vlm_latency=47.8s)
    - raw: Pick up the ceramic mug, rotate it, place it on the table, and hang it on the dark gray rack with flat back.
    - aug: Pick up the ceramic mug, rotate it, place it on the table, and hang it on the dark gray rack with flat back, the mug is in the center, the rack is on the right
  - `hanging_mug|seed=2` (vlm_latency=24.5s)
    - raw: Pick up the beige coffee mug, rotate it, place it on the table, and hang it on the dark gray rack with flat back.
    - aug: Pick up the beige coffee mug, rotate it, place it on the table, and hang it on the dark gray rack with flat back; the mug is on the left, the rack is on the right
  - `hanging_mug|seed=3` (vlm_latency=11.3s)
    - raw: Grab the cylindrical mug with handle from the table, rotate it, and set it in the center. Then hang the cylindrical mug with handle on the medium-sized rack.
    - aug: Grab the cylindrical mug with handle from the table, rotate it, set it in the center, then hang it on the medium-sized rack; the mug is on the table
  - `hanging_mug|seed=4` (vlm_latency=22.9s)
    - raw: Pick up the matte black mug, rotate it, place it on the table, and hang it on the dark gray rack with flat back.
    - aug: Pick up the matte black mug in the center, rotate it, place it on the table, and hang it on the dark gray rack with flat back on the right
  - `hanging_mug|seed=5` (vlm_latency=23.0s)
    - raw: Pick the beige coffee mug with the left arm, rotate, place it, then hang it on the dark gray rack with flat back.
    - aug: Pick the beige coffee mug with the left arm, rotate, place it, then hang it on the dark gray rack with flat back; the mug is on the left
  - `hanging_mug|seed=6` (vlm_latency=13.6s)
    - raw: Lift the cylindrical mug with handle, rotate it, put it down, then attach it to the rack with straight and slanted rods.
    - aug: Lift the cylindrical mug with handle, rotate it, put it down, then attach it to the rack with straight and slanted rods; the mug is on the left
  - `hanging_mug|seed=7` (vlm_latency=12.1s)
    - raw: Pick the beige coffee mug with the left arm, rotate, place it, then hang it on the medium-sized rack.
    - aug: Pick the beige coffee mug with the left arm, rotate, place it, then hang it on the medium-sized rack; the mug is on the left, the rack is on the right
  - `hanging_mug|seed=8` (vlm_latency=13.4s)
    - raw: Grab the dark mug with a curved handle from the table, rotate it, and set it in the center. Then hang the dark mug with a curved handle on the medium-sized rack.
    - aug: Grab the dark mug with a curved handle from the left, rotate it, set it in the center, then hang it on the medium-sized rack
  - `hanging_mug|seed=9` (vlm_latency=25.6s)
    - raw: Grab the cylindrical mug with Starbucks logo from the table, rotate it, and set it in the center. Then hang the cylindrical mug with Starbucks logo on the dark gray rack with flat back.
    - aug: Grab the cylindrical mug with Starbucks logo from the table on the left, rotate it, set it in the center, then hang it on the dark gray rack with flat back on the right
  - `hanging_mug|seed=10` (vlm_latency=49.1s)
    - raw: Grab the beige mug from the table, rotate it, and set it in the center. Then hang the beige mug on the dark gray rack with flat back.
    - aug: Grab the beige mug from the table, rotate it, set it in the center, then hang the beige mug on the dark gray rack with flat back; the mug is on the left, the rack is on the right

**vlm_cl_menu** (Closed-loop menu (M3-menu)):
- result file: MISSING
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/hanging_mug_vlm_cl_menu/hanging_mug/episode0_randomized-true_success-false.mp4`
- closed-loop summary:
  - episodes traced: 4, total VLM calls: 52 (mean 13.0/episode)
  - subtask_index histogram: {1: 16, 2: 12, 3: 12, 4: 12}
  - out-of-range subtask_index: 0, used_fallback total: 1, mean VLM latency: 3.82s
  - example trace (`hanging_mug|seed=1|mode=closed_loop`, 13 VLM calls):
    - chunk=0 status=None idx=1 aug='Use the left arm to pick up the beige coffee mug on the left side of the table.'
    - chunk=3 status=None idx=2 aug='Rotate the mug and place it in the center of the table.'
    - chunk=6 status=None idx=3 aug='Use the right arm to pick up the beige mug from the center of the table.'
    - chunk=9 status=None idx=4 aug='Hang the mug onto the rack with the right arm.'
    - chunk=12 status=None idx=1 aug='Use the left arm to pick up the beige coffee mug from the table.'
    - chunk=15 status=None idx=2 aug='Rotate the mug and place it in the center of the table.'
    - chunk=18 status=None idx=3 aug='Use the right arm to pick up the beige mug from the center of the table.'
    - chunk=21 status=None idx=4 aug='Hang the mug onto the medium-sized rack with the right arm.'
    - ... (5 more calls)

**vlm_cl_free** (Closed-loop free (M3-free)):
- result file: MISSING
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/hanging_mug_vlm_cl_free/hanging_mug/episode2_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/hanging_mug_vlm_cl_free/hanging_mug/episode0_randomized-true_success-false.mp4`
- closed-loop summary:
  - episodes traced: 4, total VLM calls: 45 (mean 11.2/episode)
  - out-of-range subtask_index: 0, used_fallback total: 0, mean VLM latency: 3.67s
  - example trace (`hanging_mug|seed=1|mode=closed_loop`, 13 VLM calls):
    - chunk=0 status=None idx=None aug='Grab the beige mug with the handle on the left side from the table.'
    - chunk=3 status=None idx=None aug='Rotate the beige mug so the handle faces right, then place it in the center of the table.'
    - chunk=6 status=None idx=None aug='Lift the beige mug from the table and hang it onto the medium-sized rack.'
    - chunk=9 status=None idx=None aug='Hang the beige mug onto the medium-sized rack using its handle.'
    - chunk=12 status=None idx=None aug='Grab the beige mug by its handle from the table.'
    - chunk=15 status=None idx=None aug='Rotate the beige mug 180 degrees so the handle faces away from you, then place it in the center of the table.'
    - chunk=18 status=None idx=None aug='Pick up the beige mug from the table center and hang it on the medium-sized rack.'
    - chunk=21 status=None idx=None aug='Grab the beige mug on the table, rotate it, set it down in the middle, then hang it onto the medium-sized rack.'
    - ... (5 more calls)


### open_microwave

**baseline** (Baseline (M1, raw)):
- success rate: **30.0%**  (3 success / 7 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_baseline/open_microwave/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_baseline/open_microwave/episode2_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_baseline/open_microwave/episode0_randomized-true_success-false.mp4`

**vlm** (Open-loop VLM (M2)):
- success rate: **60.0%**  (6 success / 4 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_vlm/open_microwave/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_vlm/open_microwave/episode0_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_vlm/open_microwave/episode2_randomized-true_success-false.mp4`
- VLM augmentations (episode_seed -> augmented_instruction):
  - `open_microwave|seed=1` (vlm_latency=17.0s)
    - raw: Move the left arm and open the kitchen microwave with slanted shape.
    - aug: Open the kitchen microwave with slanted shape using the left arm, the microwave is in the center
  - `open_microwave|seed=2` (vlm_latency=8.6s)
    - raw: Move the left arm and open the kitchen microwave with slanted shape.
    - aug: Move the left arm and open the kitchen microwave with slanted shape; the microwave is in the center
  - `open_microwave|seed=3` (vlm_latency=23.4s)
    - raw: Move the left arm and open the medium-sized microwave with angled walls.
    - aug: Move the left arm and open the medium-sized microwave with angled walls located in the center
  - `open_microwave|seed=4` (vlm_latency=8.6s)
    - raw: Open the medium gray microwave with top vents using the left arm.
    - aug: Open the medium gray microwave with top vents using the left arm, the microwave is in the center
  - `open_microwave|seed=5` (vlm_latency=26.9s)
    - raw: Move the left arm and open the medium-sized microwave with angled walls.
    - aug: Move the left arm and open the medium-sized microwave with angled walls, the microwave is in the center
  - `open_microwave|seed=6` (vlm_latency=4.5s)
    - raw: Grab the medium gray microwave with top vents's handle and pull to open it.
    - aug: Grab the medium gray microwave with top vents's handle and pull to open it, the microwave is in the center
  - `open_microwave|seed=7` (vlm_latency=7.8s)
    - raw: Locate the dark gray microwave and open it using the left arm.
    - aug: Locate the dark gray microwave in the center and open it using the left arm
  - `open_microwave|seed=8` (vlm_latency=32.7s)
    - raw: Grab the medium gray microwave with top vents's handle and pull to open it.
    - aug: Grab the medium gray microwave with top vents's handle and pull to open it the microwave is in the center
  - `open_microwave|seed=9` (vlm_latency=17.8s)
    - raw: Move the left arm and open the medium gray microwave with top vents.
    - aug: Move the left arm and open the medium gray microwave with top vents which is in the center
  - `open_microwave|seed=10` (vlm_latency=11.9s)
    - raw: Use the left arm to open the kitchen microwave with slanted shape.
    - aug: Use the left arm to open the kitchen microwave with slanted shape, the microwave is in the center

**vlm_cl_menu** (Closed-loop menu (M3-menu)):
- success rate: **40.0%**  (4 success / 6 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_vlm_cl_menu/open_microwave/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_vlm_cl_menu/open_microwave/episode0_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_vlm_cl_menu/open_microwave/episode1_randomized-true_success-false.mp4`
- closed-loop summary:
  - episodes traced: 10, total VLM calls: 156 (mean 15.6/episode)
  - subtask_index histogram: {1: 68, 2: 88}
  - out-of-range subtask_index: 0, used_fallback total: 2, mean VLM latency: 3.43s
  - example trace (`open_microwave|seed=1|mode=closed_loop`, 6 VLM calls):
    - chunk=0 status=None idx=1 aug='Use the left arm to grab the microwave door handle.'
    - chunk=3 status=None idx=2 aug='Pull the microwave door open with the left arm.'
    - chunk=6 status=None idx=1 aug='Use the left arm to grab the microwave door handle.'
    - chunk=9 status=None idx=2 aug='Pull the microwave door open with the left arm.'
    - chunk=12 status=None idx=1 aug='Use the left arm to grab the microwave door handle.'
    - chunk=15 status=None idx=2 aug='Pull the microwave door open with the left arm.'

**vlm_cl_free** (Closed-loop free (M3-free)):
- success rate: **20.0%**  (2 success / 8 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_vlm_cl_free/open_microwave/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_vlm_cl_free/open_microwave/episode1_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/open_microwave_vlm_cl_free/open_microwave/episode0_randomized-true_success-false.mp4`
- closed-loop summary:
  - episodes traced: 10, total VLM calls: 185 (mean 18.5/episode)
  - out-of-range subtask_index: 0, used_fallback total: 0, mean VLM latency: 3.57s
  - example trace (`open_microwave|seed=1|mode=closed_loop`, 21 VLM calls):
    - chunk=0 status=None idx=None aug="Move your left arm toward the microwave's left handle."
    - chunk=3 status=None idx=None aug='Grasp the black left handle of the microwave with your left gripper.'
    - chunk=6 status=None idx=None aug='Pull the black left handle of the microwave outward to open the door.'
    - chunk=9 status=None idx=None aug='Continue pulling the black left handle outward until the microwave door opens fully.'
    - chunk=12 status=None idx=None aug='Pull the black left handle outward further to open the microwave door fully.'
    - chunk=15 status=None idx=None aug='Pull the black left handle outward until the microwave door opens fully.'
    - chunk=18 status=None idx=None aug='Continue pulling the black left handle outward until the microwave door opens fully.'
    - chunk=21 status=None idx=None aug='Pull the black left handle outward further to fully open the microwave door.'
    - ... (13 more calls)


### place_can_basket

**baseline** (Baseline (M1, raw)):
- success rate: **30.0%**  (3 success / 7 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/place_can_basket_baseline/place_can_basket/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/place_can_basket_baseline/place_can_basket/episode2_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/place_can_basket_baseline/place_can_basket/episode0_randomized-true_success-false.mp4`

**vlm** (Open-loop VLM (M2)):
- success rate: **60.0%**  (6 success / 4 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/place_can_basket_vlm/place_can_basket/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/place_can_basket_vlm/place_can_basket/episode2_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/place_can_basket_vlm/place_can_basket/episode0_randomized-true_success-false.mp4`
- VLM augmentations (episode_seed -> augmented_instruction):
  - `place_can_basket|seed=1` (vlm_latency=30.8s)
    - raw: Pick the red can, put it in the yellow rectangular basket with dual grips, then lift the yellow rectangular basket with dual grips.
    - aug: Pick the red can that is on the right, put it in the yellow rectangular basket with dual grips, then lift the yellow rectangular basket with dual grips
  - `place_can_basket|seed=2` (vlm_latency=6.0s)
    - raw: Grab the small aluminum can and put it in the yellow rectangular basket with dual grips, then lift the yellow rectangular basket with dual grips.
    - aug: Grab the small aluminum can on the right and put it in the yellow rectangular basket with dual grips, then lift the yellow rectangular basket with dual grips
  - `place_can_basket|seed=3` (vlm_latency=19.2s)
    - raw: Grab the can with gold logo design, drop it into the yellow rectangular basket with dual grips, lift the yellow rectangular basket with dual grips with another arm.
    - aug: Grab the can with gold logo design on the left, drop it into the yellow rectangular basket with dual grips, lift the yellow rectangular basket with dual grips with another arm
  - `place_can_basket|seed=4` (vlm_latency=21.9s)
    - raw: Pick up the can with shiny smooth surface, place it in the medium basket with smooth texture, then lift the medium basket with smooth texture.
    - aug: Pick up the can with shiny smooth surface on the right, place it in the medium basket with smooth texture in the center, then lift the medium basket with smooth texture
  - `place_can_basket|seed=5` (vlm_latency=25.3s)
    - raw: Pick the cylindrical can with smooth surface, put it in the oval basket with medium storage space, then lift the oval basket with medium storage space.
    - aug: Pick the cylindrical can with smooth surface on the right, put it in the oval basket with medium storage space in the center, then lift the oval basket with medium storage space
  - `place_can_basket|seed=6` (vlm_latency=27.7s)
    - raw: Use the right arm to pick the can with red body, place it into the medium basket with smooth texture, and lift the basket.
    - aug: Use the right arm to pick the red can on the right, place it into the medium basket in the center, and lift the basket.
  - `place_can_basket|seed=7` (vlm_latency=46.8s)
    - raw: Use the left arm to grab the cylindrical brown drink can with white markings, put it in the medium basket with smooth texture, and lift the medium basket with smooth texture.
    - aug: Use left arm to grab the cylindrical brown drink can with white markings on left, put it in the medium smooth basket at center, and lift the basket
  - `place_can_basket|seed=8` (vlm_latency=34.2s)
    - raw: Use the right arm to pick the lightweight cylindrical metal can, drop it into the yellow basket with open slots, then lift the yellow basket with open slots.
    - aug: Use the right arm to pick the lightweight cylindrical metal can on the right, drop it into the yellow basket with open slots, then lift the yellow basket with open slots.
  - `place_can_basket|seed=9` (vlm_latency=56.3s)
    - raw: Use the right arm to pick the can with gold logo design, drop it into the oval basket with medium storage space, then lift the oval basket with medium storage space.
    - aug: Use the right arm to pick the can with gold logo design at the top, drop it into the oval basket with medium storage space in the center, then lift the oval basket with medium storage space.
  - `place_can_basket|seed=10` (vlm_latency=14.8s)
    - raw: Use the right arm to pick the small red soda can, place it into the yellow basket with open slots, and lift the basket.
    - aug: Use the right arm to pick the small red soda can, place it into the yellow basket with open slots, and lift the basket, the can is on the right, the basket is center

**vlm_cl_menu** (Closed-loop menu (M3-menu)):
- result file: MISSING
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/place_can_basket_vlm_cl_menu/place_can_basket/episode0_randomized-true_success-false.mp4`
- closed-loop summary:
  - episodes traced: 2, total VLM calls: 20 (mean 10.0/episode)
  - subtask_index histogram: {1: 7, 2: 6, 3: 7}
  - out-of-range subtask_index: 0, used_fallback total: 1, mean VLM latency: 3.72s
  - example trace (`place_can_basket|seed=1|mode=closed_loop`, 10 VLM calls):
    - chunk=0 status=None idx=1 aug='Use the left arm to pick up the red and gold soda can on the right side of the table.'
    - chunk=3 status=None idx=2 aug='Place the gold and red soda can into the yellow rectangular basket with dual grips.'
    - chunk=6 status=None idx=3 aug='Lift the yellow rectangular basket with dual grips using the right arm.'
    - chunk=9 status=None idx=1 aug='Use the left arm to pick up the gold and red soda can on the left side of the table.'
    - chunk=12 status=None idx=2 aug='Place the gold and red soda can into the yellow rectangular basket with dual grips.'
    - chunk=15 status=None idx=3 aug='Lift the yellow rectangular basket with dual grips using the right arm.'
    - chunk=18 status=None idx=1 aug='Use the left arm to pick up the red soda can with gold label on the right side of the table.'
    - chunk=21 status=None idx=2 aug='Place the gold and red soda can into the yellow rectangular basket with dual grips.'
    - ... (2 more calls)

**vlm_cl_free** (Closed-loop free (M3-free)):
- result file: MISSING
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/place_can_basket_vlm_cl_free/place_can_basket/episode1_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/place_can_basket_vlm_cl_free/place_can_basket/episode0_randomized-true_success-false.mp4`
- closed-loop summary:
  - episodes traced: 3, total VLM calls: 14 (mean 4.7/episode)
  - out-of-range subtask_index: 0, used_fallback total: 0, mean VLM latency: 4.23s
  - example trace (`place_can_basket|seed=1|mode=closed_loop`, 10 VLM calls):
    - chunk=0 status=None idx=None aug='Grab the red aluminum can on the right side of the table.'
    - chunk=3 status=None idx=None aug='Drop the red aluminum can into the yellow basket with open slots.'
    - chunk=6 status=None idx=None aug='Lift the yellow basket with open slots from the table.'
    - chunk=9 status=None idx=None aug='Grab the small aluminum can from the table.'
    - chunk=12 status=None idx=None aug='Drop the small aluminum can into the yellow basket with open slots.'
    - chunk=15 status=None idx=None aug='Lift the yellow basket with open slots from the table.'
    - chunk=18 status=None idx=None aug='Grab the small aluminum can from the table.'
    - chunk=21 status=None idx=None aug='Drop the aluminum can into the yellow basket with open slots.'
    - ... (2 more calls)


### turn_switch

**baseline** (Baseline (M1, raw)):
- success rate: **40.0%**  (4 success / 6 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_baseline/turn_switch/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_baseline/turn_switch/episode1_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_baseline/turn_switch/episode0_randomized-true_success-false.mp4`

**vlm** (Open-loop VLM (M2)):
- success rate: **40.0%**  (4 success / 6 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_vlm/turn_switch/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_vlm/turn_switch/episode0_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_vlm/turn_switch/episode1_randomized-true_success-false.mp4`
- VLM augmentations (episode_seed -> augmented_instruction):
  - `turn_switch|seed=1` (vlm_latency=23.9s)
    - raw: Press the small beige switch with angular sides using the right arm
    - aug: Press the small beige switch with angular sides using the right arm the switch is in the center
  - `turn_switch|seed=2` (vlm_latency=60.5s)
    - raw: Use the left arm to press the smooth black switch
    - aug: Use the left arm to press the smooth black switch the switch is in the center
  - `turn_switch|seed=3` (vlm_latency=10.7s)
    - raw: Click the flat gray switch with the right arm.
    - aug: Click the flat gray switch with the right arm, the switch is on the right
  - `turn_switch|seed=4` (vlm_latency=14.9s)
    - raw: Use the left arm to press the switch with circular black buttons
    - aug: Use the left arm to press the switch with circular black buttons located in the center
  - `turn_switch|seed=5` (vlm_latency=26.6s)
    - raw: Click the gray switch with slanted sides with the right arm.
    - aug: Click the gray switch with slanted sides with the right arm, the switch is on the right
  - `turn_switch|seed=6` (vlm_latency=19.3s)
    - raw: Use the left arm to press the simple toggle switch.
    - aug: Use the left arm to press the simple toggle switch the switch is on the right
  - `turn_switch|seed=7` (vlm_latency=10.7s)
    - raw: Click the switch with two metal prongs with the left arm
    - aug: Click the switch with two metal prongs with the left arm, the switch is in the center
  - `turn_switch|seed=8` (vlm_latency=17.3s)
    - raw: Click the smooth black switch with red lever with the left arm.
    - aug: Click the smooth black switch with red lever with the left arm, the switch is on the right
  - `turn_switch|seed=9` (vlm_latency=23.9s)
    - raw: Activate the switch at the small beige switch with angular sides
    - aug: Activate the small beige switch with angular sides, the switch is on the right
  - `turn_switch|seed=10` (vlm_latency=5.7s)
    - raw: Click the beige switch with darker brown top with the right arm
    - aug: Click the beige switch with darker brown top with the right arm, the switch is on the right

**vlm_cl_menu** (Closed-loop menu (M3-menu)):
- success rate: **70.0%**  (7 success / 3 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_vlm_cl_menu/turn_switch/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_vlm_cl_menu/turn_switch/episode1_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_vlm_cl_menu/turn_switch/episode0_randomized-true_success-false.mp4`
- closed-loop summary:
  - episodes traced: 10, total VLM calls: 28 (mean 2.8/episode)
  - subtask_index histogram: {1: 28}
  - out-of-range subtask_index: 0, used_fallback total: 0, mean VLM latency: 3.28s
  - example trace (`turn_switch|seed=1|mode=closed_loop`, 6 VLM calls):
    - chunk=0 status=None idx=1 aug='Press the flat gray switch on the table with the left arm.'
    - chunk=3 status=None idx=1 aug='Press the flat gray switch with the left arm, the one mounted on the white rectangular base near the center of the table.'
    - chunk=6 status=None idx=1 aug='Press the flat gray switch with the left arm, the one mounted on the white rectangular plate near the center of the table.'
    - chunk=9 status=None idx=1 aug='Press the flat gray switch with the left arm, the one standing upright near the center of the table.'
    - chunk=12 status=None idx=1 aug='Press the flat gray switch with the left arm, the one on the right side of the table near the orange can.'
    - chunk=15 status=None idx=1 aug='Press the flat gray switch with the left arm, the one mounted on the white bracket near the right edge of the table.'

**vlm_cl_free** (Closed-loop free (M3-free)):
- success rate: **70.0%**  (7 success / 3 fail) `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_vlm_cl_free/turn_switch/_result_random.txt`
- sample success video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_vlm_cl_free/turn_switch/episode1_randomized-true_success-true.mp4`
- sample failure video: `evaluate_results/robotwin/robotwin_uncond_3cam_384/turn_switch_vlm_cl_free/turn_switch/episode0_randomized-true_success-false.mp4`
- closed-loop summary:
  - episodes traced: 10, total VLM calls: 31 (mean 3.1/episode)
  - out-of-range subtask_index: 0, used_fallback total: 0, mean VLM latency: 3.51s
  - example trace (`turn_switch|seed=1|mode=closed_loop`, 6 VLM calls):
    - chunk=0 status=None idx=None aug='Move right arm toward the white switch with circular black buttons on the table'
    - chunk=3 status=None idx=None aug='Click the right circular black button on the white switch using the right arm'
    - chunk=6 status=None idx=None aug='Click the right circular black button on the white switch using the right arm'
    - chunk=9 status=None idx=None aug='Click the right circular black button on the white switch with the right arm'
    - chunk=12 status=None idx=None aug='Click the left circular black button on the white switch with the right arm'
    - chunk=15 status=None idx=None aug='Click the right circular black button on the white switch with the right arm'

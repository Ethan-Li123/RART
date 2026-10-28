# Test examples

Five two-object motion-transfer examples are included:

| Case | Target objects |
| --- | --- |
| case_01 | Two people |
| case_02 | Fantastical creature and owl |
| case_03 | Capybara and man |
| case_04 | Elephant and tiger |
| case_05 | Cartoon dog and koala |

Each case contains two reference videos, two reference masks, one target image,
and two target masks. The video inputs contain 49 frames.

`inference.csv` lists the inputs and output names. All media paths are relative
to the project root. Reference videos and reference masks use comma-separated
lists; target masks use `+`; object descriptions (`persons`) and motions
(`actions`) use `|`. The object order must agree across these fields.
`ref_videos_clip_starts` gives one start frame per reference video, and
`target_image_frame_idx` selects the target frame when the target is a video.

Run inference from the project root and select this CSV with
`--inference_csv data/RART_test_data/inference.csv`.

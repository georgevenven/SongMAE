European starling annotations are full-recording labels from extracted wav folders:
`/media/george-vengrovski/disk2/raw_data/european_starling/audio_files/3237218`

Excluded individuals:

- `B335`
- `B336`
- `B337`
- `B338`

These four birds were previously excluded in individual_id artifacts as
`drop_B335_B336_B337_B338`.

Filenames must stay prefixed as `<bird_id>__<wav_name>`. The raw wav basenames
are not unique across birds, so unprefixed labels collide.

No unit-level label table was found in the local raw data folder.

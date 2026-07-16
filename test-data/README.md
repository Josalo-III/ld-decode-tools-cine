# Integration test fixtures

`ntsc/ggv-1069-multiburst.tbc` is a 12-frame cut from Pioneer NTSC
reference disc GGV-1069, CAV picture numbers 1000 through 1011. The disc's
published programme identifies pictures 901 through 1800 as a monochrome
multiburst signal. The cut was made with:

```text
ld-tbc-trim --cav 1000 --length 12 GGV-1069.tbc ggv-1069-multiburst.tbc
```

Its nominally monochrome high-frequency bands make it useful for detecting
cross-colour leakage as well as checking the ordinary pre-TBC pipeline.

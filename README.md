# Tetrachromatic: ML utils for the hacker with 4D vision.

This repo contains self-contained utilities and snippets that I often find myself reaching for in machine learning projects.

This repo is intentionally **not** pip-installable. Each utility is implemented as a single file which you can copy-paste into your projects. Documentation primarily takes the form of a long docstring at the beginning of each file. I've also tried my best to include the most relevant references wherever possible.

## Utils

- **edm2/** (WIP) JAX reimplementation of the Karras et al. EDM2 UNet architecture.
- **schedulers/** A simple collection of stateless learning rate schedulers.

## On project versioning and naming

There's no logic behind the project name. I wanted to call it 'ml-bazaar', but that was already taken. After clicking through synonyms-of-synonyms a few times, the word _polychromatic_ came up and I thought it sounded cool. This reminded me of a KGLW song called tetrachromacy, which, I think is fair to say, is even cooler.

Similarly, this project uses a variant of [sentimental versioning](https://github.com/dominictarr/sentimental-versioning/) that I came up with whilst hungover. In an effort to establish this codebase as the fastest growing project in the universe, major versions will be numbered following the [busy-beaver](https://en.wikipedia.org/wiki/Busy_beaver) sequence.

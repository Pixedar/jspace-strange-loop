# License scope

The current author-owned software is released under **GNU Affero General Public License version 3 only (AGPL-3.0-only)**, in [LICENSE](LICENSE). This includes the scientific code and the interactive viewer. The license preserves modification and redistribution rights and requires corresponding source under its applicable copyleft conditions, including its network-interaction provision for modified software. It permits commercial use; it is not a noncommercial or field-of-use restriction.

New author-owned text, figures and numeric observations in this release are **CC BY-SA 4.0**, in [LICENSE-DATA](LICENSE-DATA). Credit and ShareAlike apply to covered adaptations. The Hugging Face release documents the scope of the same code/data split.

Earlier releases remain available under the licenses under which they were published. This change does not revoke already granted MIT or CC BY 4.0 permissions. The prior license texts are preserved with the [archived release account](history/).

Third-party material retains its own terms:

- Qwen model weights remain in their original repositories under their model licenses. The released small interface checkpoints do not contain the frozen base-model weights.
- The Qwen2.5-Omni-3B follow-up uses the upstream Qwen Research license. Publishing this experiment’s code and observations does not relicense the model.
- Transformers and other dependencies retain their respective licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- The combined external text corpus is not redistributed here. WikiText, AG News, TinyStories and DialogSum have differing upstream terms, including noncommercial restrictions in DialogSum. `self_reference/corpus_sources.json` identifies them. The constructor downloads those sources separately; this repository’s licenses do not replace their terms.
- Earlier thinking/instruction datasets and Jacobian lenses retain their upstream terms and attribution. Their inclusion in an earlier experiment does not make them newly licensed under AGPL or CC BY-SA.

Author: Wiktor Tomasik, 2026.

Authoritative license texts: [GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The AGPL is an [OSI-approved open-source license](https://opensource.org/license/agpl-3-0).

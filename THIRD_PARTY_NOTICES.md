# Third-Party Notices

This repository includes third-party components that remain under their own licenses. OmniSVG itself is still licensed under the terms in the top-level [LICENSE](LICENSE).

## Cairo

- Component: Cairo 2D graphics library
- Path: `third_party/cairo`
- Upstream: `https://gitlab.freedesktop.org/cairo/cairo.git`
- Pinned revision: `4ca0d581cbbbc3d534da3b0d10e8271cd0336c37`
- License choice used by this repository: `LGPL-2.1`
- Alternative upstream license: `MPL-1.1`

Cairo is shipped as a git submodule so the original source, notices, and history stay intact. The submodule contains the upstream license texts at:

- `third_party/cairo/COPYING`
- `third_party/cairo/COPYING-LGPL-2.1`
- `third_party/cairo/COPYING-MPL-1.1`

OmniSVG does not bundle Cairo DLLs in this repository by default. On Windows, the recommended setup is to install a system Cairo runtime and load it dynamically at run time. This keeps the library replaceable by the end user, which is the safer compliance path for LGPL-linked deployments.

If you distribute a build that includes Cairo binaries, keep the Cairo license text with the distribution, keep the Cairo source or an equivalent source offer available, and do not block users from replacing the Cairo library with a compatible modified build.

This notice is provided for engineering convenience and is not legal advice.

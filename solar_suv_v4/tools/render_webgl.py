#!/usr/bin/env python3
"""Offline WebGL renderer for the figures (pip install playwright; uses the local Chromium,
no network).  Each object is a triangle mesh with a colour, optional alpha (ghosted shells)
and an optional clip plane (cutaways).  Crease-aware smooth normals, two lights + rim light,
Blinn specular, 2x supersampling.

    from render_webgl import Scene
    s = Scene(1600, 900)
    s.add(verts, faces, (0.9, 0.9, 0.92), alpha=1.0, clip=None, spec=0.4)
    s.camera(eye, target, up=(0, 0, 1), fov=30)          # or ortho=half_height_mm
    s.render("out.png")
"""
import glob, io, json, math, os
import numpy as np
from PIL import Image

CHROME = (sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")) or [None])[-1]

_HTML = r"""<!doctype html><html><body style="margin:0;background:#fff">
<canvas id="c"></canvas><script>
async function run(spec){
  const W = spec.w, H = spec.h, cv = document.getElementById('c');
  cv.width = W; cv.height = H;
  const gl = cv.getContext('webgl2', {antialias:false, preserveDrawingBuffer:true, alpha:false});
  const vs = `#version 300 es
  in vec3 p; in vec3 n; uniform mat4 mvp; out vec3 vn; out vec3 vp;
  void main(){ vn = n; vp = p; gl_Position = mvp*vec4(p,1.0); }`;
  const fs = `#version 300 es
  precision highp float; in vec3 vn; in vec3 vp; out vec4 o;
  uniform vec3 col; uniform float alpha; uniform vec4 clip; uniform vec3 eye; uniform float spec;
  uniform vec3 L1; uniform vec3 L2;
  void main(){
    if (dot(clip.xyz, vp) > clip.w) discard;
    vec3 N = normalize(vn); vec3 V = normalize(eye - vp);
    if (!gl_FrontFacing) N = -N;
    float d1 = max(dot(N, L1), 0.0), d2 = max(dot(N, L2), 0.0);
    float hemi = 0.5 + 0.5*N.z;
    vec3 Hh = normalize(L1 + V);
    float s = pow(max(dot(N, Hh), 0.0), 60.0)*spec;
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0)*0.18;
    vec3 c = col*(0.30 + 0.20*hemi + 0.62*d1 + 0.22*d2) + vec3(s + rim);
    if (!gl_FrontFacing) c *= 0.55;
    o = vec4(pow(c, vec3(1.0/1.1)), alpha);
  }`;
  function sh(t, s){ const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x);
    if(!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(x); return x; }
  const pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(pr); gl.useProgram(pr);
  const U = n => gl.getUniformLocation(pr, n);
  gl.viewport(0, 0, W, H);
  const bg = spec.bg; gl.clearColor(bg[0], bg[1], bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.uniformMatrix4fv(U('mvp'), false, new Float32Array(spec.mvp));
  gl.uniform3fv(U('eye'), spec.eye); gl.uniform3fv(U('L1'), spec.L1); gl.uniform3fv(U('L2'), spec.L2);
  const buf = await (await fetch('/data.bin')).arrayBuffer();
  const order = spec.objs.map((o, i) => i).sort((a, b) => (spec.objs[a].alpha < 1) - (spec.objs[b].alpha < 1));
  for (const i of order){
    const o = spec.objs[i];
    const P = new Float32Array(buf, o.off, o.n*3), Nn = new Float32Array(buf, o.off + o.n*12, o.n*3);
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    for (const [name, arr] of [['p', P], ['n', Nn]]){
      const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(pr, name); gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    }
    gl.uniform3fv(U('col'), o.col); gl.uniform1f(U('alpha'), o.alpha); gl.uniform4fv(U('clip'), o.clip);
    gl.uniform1f(U('spec'), o.spec);
    if (o.alpha < 1){ gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); }
    else { gl.disable(gl.BLEND); gl.depthMask(true); }
    gl.drawArrays(gl.TRIANGLES, 0, o.n);
    gl.deleteVertexArray(vao);
  }
  gl.finish();
  return cv.toDataURL('image/png');
}
</script></body></html>"""


def corner_normals(v, f, crease=35.0):
    """Unindexed positions and crease-aware smooth normals (float32, (3*len(f), 3) each)."""
    v = np.asarray(v, np.float64); f = np.asarray(f, np.int64)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20)
    cn = vn[f]                                            # (m, 3, 3)
    ok = np.einsum("mkj,mj->mk", cn, fn) > math.cos(math.radians(crease))
    cn = np.where(ok[..., None], cn, fn[:, None, :])
    return tri.reshape(-1, 3).astype(np.float32), cn.reshape(-1, 3).astype(np.float32)


def _look(eye, target, up):
    eye, target, up = (np.asarray(a, float) for a in (eye, target, up))
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    r = np.cross(fwd, up); r /= np.linalg.norm(r)
    u = np.cross(r, fwd)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = r, u, -fwd
    M[:3, 3] = -M[:3, :3] @ eye
    return M


class Scene:
    def __init__(self, w=1600, h=1000, ss=2, bg=(1, 1, 1)):
        self.w, self.h, self.ss, self.bg = w, h, ss, bg
        self.objs, self.blobs = [], []
        self.off = 0

    def add(self, v, f, col, alpha=1.0, clip=None, spec=0.35, crease=35.0):
        if len(f) == 0:
            return
        P, N = corner_normals(v, f, crease)
        self.blobs += [P.tobytes(), N.tobytes()]
        self.objs.append(dict(off=self.off, n=len(P), col=list(map(float, col)), alpha=float(alpha),
                              clip=list(clip) if clip is not None else [0, 0, 0, 1e9], spec=float(spec)))
        self.off += P.nbytes + N.nbytes
        self._pts = np.vstack([getattr(self, "_pts", np.zeros((0, 3))), np.asarray(v)[::max(1, len(v)//2000)]])

    def camera(self, eye, target, up=(0, 0, 1), fov=30.0, ortho=None, near=None, far=None):
        fwd = np.asarray(target, float) - np.asarray(eye, float)
        if np.linalg.norm(np.cross(fwd, up)) < 1e-6*np.linalg.norm(fwd):     # looking straight down
            up = (0, 1, 0)
        self.eye, self.target, self.up, self.fov, self.ortho = eye, target, up, fov, ortho
        self.near, self.far = near, far

    def render(self, path=None):
        from playwright.sync_api import sync_playwright
        W, H = self.w*self.ss, self.h*self.ss
        V = _look(self.eye, self.target, self.up)
        d = np.linalg.norm(np.asarray(self.eye, float) - np.asarray(self.target, float))
        pts = (V @ np.c_[self._pts, np.ones(len(self._pts))].T)[:3].T
        near = self.near or max(1e-3, -pts[:, 2].max() - 0.2*d) if self.ortho is None else 0.1
        far = self.far or (-pts[:, 2].min() + 0.2*d + 1.0)
        a = W/H
        if self.ortho:
            t = self.ortho; r = t*a
            P = np.array([[1/r, 0, 0, 0], [0, 1/t, 0, 0], [0, 0, -2/(far - near), -(far + near)/(far - near)], [0, 0, 0, 1]])
            if self.near is None:
                near, far = -pts[:, 2].max() - 10, -pts[:, 2].min() + 10
                P[2, 2], P[2, 3] = -2/(far - near), -(far + near)/(far - near)
        else:
            near = max(near, 1e-2)
            fy = 1/math.tan(math.radians(self.fov)/2)
            P = np.array([[fy/a, 0, 0, 0], [0, fy, 0, 0], [0, 0, (far + near)/(near - far), 2*far*near/(near - far)], [0, 0, -1, 0]])
        mvp = (P @ V).T.reshape(-1).tolist()                  # column-major for WebGL
        fwd = np.asarray(self.target, float) - np.asarray(self.eye, float); fwd /= np.linalg.norm(fwd)
        r = np.cross(fwd, self.up); r /= np.linalg.norm(r); u = np.cross(r, fwd)
        L1 = -0.55*fwd + 0.55*u - 0.45*r; L1 /= np.linalg.norm(L1)
        L2 = -0.5*fwd - 0.2*u + 0.7*r; L2 /= np.linalg.norm(L2)
        spec = dict(w=W, h=H, mvp=mvp, eye=list(map(float, self.eye)), L1=L1.tolist(), L2=L2.tolist(),
                    objs=self.objs, bg=list(self.bg))
        import http.server, socketserver, tempfile, threading
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "data.bin"), "wb") as fh:
            for b in self.blobs:
                fh.write(b)
        with open(os.path.join(tmp, "index.html"), "w") as fh:
            fh.write(_HTML)
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=tmp, **k)
        http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
        srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        with sync_playwright() as p:
            br = p.chromium.launch(executable_path=CHROME, args=["--use-gl=angle", "--use-angle=swiftshader",
                                                                   "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
                                                                   "--disable-dev-shm-usage", "--no-sandbox", "--no-proxy-server",
                                                                   "--js-flags=--max-old-space-size=8192"])
            pg = br.new_page(viewport=dict(width=min(W, 4000), height=min(H, 4000)))
            pg.goto(f"http://127.0.0.1:{port}/index.html")
            url = pg.evaluate("spec => run(spec)", spec)
            br.close()
        srv.shutdown(); srv.server_close()
        import base64, shutil
        shutil.rmtree(tmp, ignore_errors=True)
        img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
        if self.ss > 1:
            img = img.resize((self.w, self.h), Image.LANCZOS)
        if path:
            img.save(path)
        return img

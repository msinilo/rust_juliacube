extern crate minifb;
extern crate rayon;

use std::time::Instant;
use minifb::{Key, WindowOptions, Window};
use rayon::prelude::*;
use std::thread;
use std::slice;
use std::sync::mpsc::channel;

use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;

const RESOLUTION : usize    = 768;
const TEX_RESOLUTION : usize = 768;
const CHUNK_SIZE : usize    = (TEX_RESOLUTION / 16);
const FRES : f32            = TEX_RESOLUTION as f32;
const HALF_RES : f32        = FRES * 0.5;
const MAX_ITER : u32 = 64;

type Rfloat = f32;
type Rfixed = i32;

#[derive(Copy, Clone, Debug)]
struct Vector2 {
    x: Rfloat,
    y: Rfloat,
}
struct Point2 {
    x: Rfixed,
    y: Rfixed,
}

#[derive(Copy, Clone, Debug)]
struct Vector3 {
    x: Rfloat,
    y: Rfloat,
    z: Rfloat,
}
struct Vector4 {
    x: Rfloat,
    y: Rfloat,
    z: Rfloat,
    w: Rfloat,
}

#[derive(Copy, Clone)]
struct Matrix4 {
    data: [[Rfloat; 4]; 4],
}

#[derive(Copy, Clone)]
struct Vertex {
    pos: Vector3,
    //normal: Vector3,
    //color : Vector3
    uv: Vector2,
}

impl Default for Vertex {
    fn default() -> Vertex {
        Vertex {
            pos: Vector3::new(0.0, 0.0, 0.0),
            //normal: Vector3::new(0.0, 0.0, 0.0),
            uv: Vector2::new(0.0, 0.0),
        }
    }
}

struct Triangle {
    v0: usize,
    v1: usize,
    v2: usize,
}

const ZERO_VECTOR: Vector3 = Vector3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};
const ZERO_VECTOR2: Vector2 = Vector2 { x: 0.0, y: 0.0 };

impl Vector2 {
    fn new(vx: Rfloat, vy: Rfloat) -> Vector2 {
        Vector2 { x: vx, y: vy }
    }
}

impl Vector3 {
    fn new(vx: Rfloat, vy: Rfloat, vz: Rfloat) -> Vector3 {
        Vector3 {
            x: vx,
            y: vy,
            z: vz,
        }
    }
}

impl Sub for Vector3 {
    type Output = Vector3;
    fn sub(self: Vector3, b: Vector3) -> Vector3 {
        Vector3 {
            x: self.x - b.x,
            y: self.y - b.y,
            z: self.z - b.z,
        }
    }
}
impl Add for Vector3 {
    type Output = Vector3;
    fn add(self: Vector3, b: Vector3) -> Vector3 {
        Vector3 {
            x: self.x + b.x,
            y: self.y + b.y,
            z: self.z + b.z,
        }
    }
}
impl Div<Rfloat> for Vector3 {
    type Output = Vector3;
    fn div(self: Vector3, s: Rfloat) -> Vector3 {
        Vector3 {
            x: self.x / s,
            y: self.y / s,
            z: self.z / s,
        }
    }
}
impl Mul<Rfloat> for Vector3 {
    type Output = Vector3;
    fn mul(self: Vector3, s: Rfloat) -> Vector3 {
        Vector3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl Matrix4 {
    fn new(
        m00: Rfloat,
        m01: Rfloat,
        m02: Rfloat,
        m03: Rfloat,
        m10: Rfloat,
        m11: Rfloat,
        m12: Rfloat,
        m13: Rfloat,
        m20: Rfloat,
        m21: Rfloat,
        m22: Rfloat,
        m23: Rfloat,
        m30: Rfloat,
        m31: Rfloat,
        m32: Rfloat,
        m33: Rfloat,
    ) -> Matrix4 {
        Matrix4 {
            data: [
                [m00, m01, m02, m03],
                [m10, m11, m12, m13],
                [m20, m21, m22, m23],
                [m30, m31, m32, m33],
            ],
        }
    }
}

macro_rules! mtx_elem {
    ($a : ident, $b : ident, $r : expr, $c : expr) => {
        $a.data[$r][0] * $b.data[0][$c]
            + $a.data[$r][1] * $b.data[1][$c]
            + $a.data[$r][2] * $b.data[2][$c]
            + $a.data[$r][3] * $b.data[3][$c]
    };
}

impl Mul for Matrix4 {
    type Output = Matrix4;
    fn mul(self: Matrix4, b: Matrix4) -> Matrix4 {
        Matrix4::new(
            mtx_elem!(self, b, 0, 0),
            mtx_elem!(self, b, 0, 1),
            mtx_elem!(self, b, 0, 2),
            mtx_elem!(self, b, 0, 3),
            mtx_elem!(self, b, 1, 0),
            mtx_elem!(self, b, 1, 1),
            mtx_elem!(self, b, 1, 2),
            mtx_elem!(self, b, 1, 3),
            mtx_elem!(self, b, 2, 0),
            mtx_elem!(self, b, 2, 1),
            mtx_elem!(self, b, 2, 2),
            mtx_elem!(self, b, 2, 3),
            mtx_elem!(self, b, 3, 0),
            mtx_elem!(self, b, 3, 1),
            mtx_elem!(self, b, 3, 2),
            mtx_elem!(self, b, 3, 3),
        )
    }
}

fn transform_point4(v: &Vector3, m: &Matrix4) -> Vector4 {
    Vector4 {
        x: v.x * m.data[0][0] + v.y * m.data[1][0] + v.z * m.data[2][0] + m.data[3][0],
        y: v.x * m.data[0][1] + v.y * m.data[1][1] + v.z * m.data[2][1] + m.data[3][1],
        z: v.x * m.data[0][2] + v.y * m.data[1][2] + v.z * m.data[2][2] + m.data[3][2],
        w: v.x * m.data[0][3] + v.y * m.data[1][3] + v.z * m.data[2][3] + m.data[3][3],
    }
}

fn perspective_projection_matrix(
    vertical_fov_deg: Rfloat,
    aspect: Rfloat,
    near: Rfloat,
    far: Rfloat,
) -> Matrix4 {
    let tfov = (vertical_fov_deg.to_radians() / 2.0).tan();
    let m00 = 1.0 / (aspect * tfov);
    let m11 = 1.0 / tfov;
    let oo_fmn = 1.0 / (far - near);
    let m22 = -(far + near) * oo_fmn;
    let m23 = -(2.0 * far * near) * oo_fmn;

    Matrix4::new(
        m00, 0.0, 0.0, 0.0, 
        0.0, m11, 0.0, 0.0, 
        0.0, 0.0, m22, m23, 
        0.0, 0.0, -1.0, 0.0,
    )
}

fn viewport_matrix(w: Rfloat, h: Rfloat) -> Matrix4 {
    let wh = w * 0.5;
    let hh = h * 0.5;
    Matrix4::new(
        wh, 0.0, 0.0, 0.0, 
        0.0, hh, 0.0, 0.0, 
        0.0, 0.0, 1.0, 0.0, 
        wh, hh, 0.0, 1.0,
    )
}

// Blob -1600
fn view_matrix(cam_dist: Rfloat) -> Matrix4 {
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -cam_dist, 1.0,
    )
}

fn rotx_matrix(degs: Rfloat) -> Matrix4 {
    let rads = degs.to_radians();
    let c = rads.cos();
    let s = rads.sin();

    Matrix4::new(
        1.0, 0.0, 0.0, 0.0, 
        0.0, c, -s, 0.0, 
        0.0, s, c, 0.0, 
        0.0, 0.0, 0.0, 1.0,
    )
}
fn roty_matrix(degs: Rfloat) -> Matrix4 {
    let rads = degs.to_radians();
    let c = rads.cos();
    let s = rads.sin();

    Matrix4::new(
        c, 0.0, s, 0.0, 
        0.0, 1.0, 0.0, 0.0, 
        -s, 0.0, c, 0.0, 
        0.0, 0.0, 0.0, 1.0,
    )
}
fn rotz_matrix(degs: Rfloat) -> Matrix4 {
    let rads = degs.to_radians();
    let c = rads.cos();
    let s = rads.sin();

    Matrix4::new(
        c, -s, 0.0, 0.0, 
        s, c, 0.0, 0.0, 
        0.0, 0.0, 1.0, 0.0, 
        0.0, 0.0, 0.0, 1.0,
    )
}

fn min3<T: PartialOrd>(a: T, b: T, c: T) -> T {
    if a < b {
        if a < c {
            a
        } else {
            c
        }
    } else if b < c {
        b
    } else {
        c
    }
}
fn max3<T: PartialOrd>(a: T, b: T, c: T) -> T {
    if a > b {
        if a > c {
            a
        } else {
            c
        }
    } else if b > c {
        b
    } else {
        c
    }
}
fn minxy(
    (x0, y0): (Rfixed, Rfixed),
    (x1, y1): (Rfixed, Rfixed),
    (x2, y2): (Rfixed, Rfixed),
) -> (Rfixed, Rfixed) {
    let minx = min3(x0, x1, x2);
    let miny = min3(y0, y1, y2);
    ((minx as Rfixed + 0xF) >> 4, (miny as Rfixed + 0xF) >> 4)
}
fn maxxy(
    (x0, y0): (Rfixed, Rfixed),
    (x1, y1): (Rfixed, Rfixed),
    (x2, y2): (Rfixed, Rfixed),
) -> (Rfixed, Rfixed) {
    let maxx = max3(x0, x1, x2);
    let maxy = max3(y0, y1, y2);
    ((maxx as Rfixed + 0xF) >> 4, (maxy as Rfixed + 0xF) >> 4)
}

fn as_fixed_pt(x: Rfloat) -> Rfixed {
    (16.0 * x) as Rfixed
}

fn orient2d(a: &Point2, b: &Point2, c: &Point2) -> Rfixed {
    ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))
}
// deltas are a bit confusing, but that's because we want to re-use edge functions
// dx is end-start (ie. >0 if going "right")
// dy is start-end (ie. >0 if going "up")
fn is_edge_top_left(dx : Rfixed, dy : Rfixed) -> bool {
    dy > 0 || (dy == 0 && dx > 0)
}

// We do not need z-buffer for a simple cube
fn draw_triangle(
    v0: &Vertex,
    v1: &Vertex,
    v2: &Vertex,
    //r: u32,
    //g: u32,
    //b: u32,
    win_min: &Point2,
    win_max: &Point2,
    buffer: &mut [u32],
    //depth_buffer: &mut [f32],
    texture: &[u32],
) {
    let x0 = as_fixed_pt(v0.pos.x);
    let y0 = as_fixed_pt(v0.pos.y);
    let x1 = as_fixed_pt(v1.pos.x);
    let y1 = as_fixed_pt(v1.pos.y);
    let x2 = as_fixed_pt(v2.pos.x);
    let y2 = as_fixed_pt(v2.pos.y);

    let p0 = Point2 { x: x0, y: y0 };
    let p1 = Point2 { x: x1, y: y1 };
    let p2 = Point2 { x: x2, y: y2 };

    // Clamp to bounding box
    let (mut minx, mut miny) = minxy((x0, y0), (x1, y1), (x2, y2));
    let (mut maxx, mut maxy) = maxxy((x0, y0), (x1, y1), (x2, y2));

    if maxx <= minx || maxy <= miny {
        return;
    }

    // Backface culling
    let tri_a2 = orient2d(&p0, &p1, &p2);
    if tri_a2 <= 0 {
        return;
    }

    if minx < win_min.x {
        minx = win_min.x
    };
    if miny < win_min.y {
        miny = win_min.y
    };
    if maxx > win_max.x {
        maxx = win_max.x
    };
    if maxy > win_max.y {
        maxy = win_max.y
    };

    let mut fb_offset = (miny as usize) * RESOLUTION;
    let inv_tri_a2 = 1.0 / tri_a2 as f32;

    let dx10 = x1 - x0;
    let dy01 = y0 - y1;
    let dx21 = x2 - x1;
    let dy12 = y1 - y2;
    let dx02 = x0 - x2;
    let dy20 = y2 - y0;
    
    // Precalc deltas to save some multiplications
    // in the inner loop
    let u10 = v1.uv.x - v0.uv.x;
    let u20 = v2.uv.x - v0.uv.x;
    let v10 = v1.uv.y - v0.uv.y;
    let v20 = v2.uv.y - v0.uv.y;
    
    // Edge function for v0v1
    // (v0y-v1y)px + (v1x-v0x)py + (v0v1y - v0yv1x)
    //   A             B            C
    // A, B & C are constants

    // -1 turns >= 0 test into > 0 
    let bias0 = if is_edge_top_left(dx10, dy01) { 0 } else { -1 };
    let bias1 = if is_edge_top_left(dx21, dy12) { 0 } else { -1 };
    let bias2 = if is_edge_top_left(dx02, dy20) { 0 } else { -1 };
    
    let c0 = x0 * y1 - y0 * x1 + bias0;
    let c1 = x1 * y2 - y1 * x2 + bias1;
    let c2 = x2 * y0 - y2 * x0 + bias2;

    let mut e0_start = dy01 * (minx << 4) + dx10 * (miny << 4) + c0;
    let mut e1_start = dy12 * (minx << 4) + dx21 * (miny << 4) + c1;
    let mut e2_start = dy20 * (minx << 4) + dx02 * (miny << 4) + c2;
    
    let fp_dx10 = dx10 << 4; // A
    let fp_dy01 = dy01 << 4; // B
    
    let fp_dx21 = dx21 << 4; // A
    let fp_dy12 = dy12 << 4; // B
    
    let fp_dx02 = dx02 << 4; // A
    let fp_dy20 = dy20 << 4; // B

    for _ty in miny..maxy {
        let mut e0x = e0_start;
        let mut e1x = e1_start;
        let mut e2x = e2_start;
        for tx in minx..maxx {
            if e0x | e1x | e2x >= 0 {
                let w0 = (e0x - bias0) as f32 * inv_tri_a2;
                //let w1 = (e1x - bias1) as f32 * inv_tri_a2;
                let w2 = (e2x - bias2) as f32 * inv_tri_a2;

                let pixel_offset = fb_offset + (tx as usize);

                //let pz = v0.pos.z + z10 * w2 + z20 * w0;
                //let d = depth_buffer[pixel_offset];
                //if d < pz {
                    //depth_buffer[pixel_offset] = pz;

                    let pu = v0.uv.x + u10 * w2 + u20 * w0;
                    let pv = v0.uv.y + v10 * w2 + v20 * w0;
                    let tr = texture[(pv as usize * TEX_RESOLUTION + pu as usize)];
                    buffer[pixel_offset] = tr;//0xFFFFFFFF;
                //}
            }

            e0x += fp_dy01;
            e1x += fp_dy12;
            e2x += fp_dy20;
        }
        e0_start += fp_dx10;
        e1_start += fp_dx21;
        e2_start += fp_dx02;
        fb_offset += RESOLUTION;
    }
}

//More or less
// https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia
fn get_color(iteration : u32) -> u32 {
    if iteration >= MAX_ITER {
        return 0;
    }
    
    let factor = (iteration as f32 / MAX_ITER as f32).sqrt();
    
    let colors = [
        66,  30,  15, // # brown 3
        25,   7,  26,// # dark violett
        9,  1,  47,// # darkest blue
        4,   4,  73,// # blue 5
        0,   7, 100,// # blue 4
        12,  44, 138,// # blue 3
        24,  82, 177,// # blue 2
        57, 125, 209,// # blue 1
        134, 181, 229,// # blue 0
        211, 236, 248,// # lightest blue
        241, 233, 191,// # lightest yellow
        248, 201,  95,// # light yellow
        255, 170,   0,// # dirty yellow
        204, 128,   0,// # brown 0
        153,  87,   0,// # brown 1
        106,  52,   3 ];// # brown 2
    let idx = ((factor * (colors.len() / 3 - 1) as f32) as usize) * 3;
    
    (0xFF << 24) | (colors[idx] << 16) | (colors[idx + 1] << 8) | colors[idx + 2]
}

fn get_color2(iteration : u32) -> u32 {
    if iteration >= MAX_ITER {
        return 0;
    }
    if iteration < MAX_ITER/8 {
        return (0xFF << 24) | (7 << 8) | 100;
    }
    if iteration < MAX_ITER / 4 {
        return (0xFF << 24) | (32 << 16) | (107 << 8) | 203;
    }
    if iteration < MAX_ITER * 3 / 8 {
        return (0xFF << 24) | (237 << 16) | (255 << 8) | 255;
    }
    if iteration < MAX_ITER / 2 {
        return (0xFF << 24) | (107 << 16) | (15 << 8) | 167;
    }
    if iteration < MAX_ITER * 5 / 8 {
        return (0xFF << 24) | (57 << 16) | (158 << 8) | 16;
    }
    if iteration < MAX_ITER * 6 / 8 {
        return (0xFF << 24) | (57 << 16) | (158 << 8) | 16;
    }
    if iteration < MAX_ITER * 7 / 8 {
        return (0xFF << 24) | (157 << 16) | (37 << 8) | 200;
    }
    
    (0xFF << 24) | (2 << 16)
}

fn mandelbrot(x_start : f32, y_start : f32) -> u32 {
    let mut x = x_start;
    let mut y = y_start;
    
    let mut iteration = 0;
    while x*x+y*y <= 4.0 && iteration < MAX_ITER {
            let x_new = x*x - y*y + x_start;
            y = 2.0*x*y + y_start;
            x = x_new;
            iteration += 1;
    }
    iteration
}

fn mandelbrot_chunk(zoom : f32, x_off : f32, y_off : f32, color_fn : impl Fn(u32)->u32, buffer : &mut[u32], offset : usize) {
    let x_start = offset % RESOLUTION;
    let y_start = offset / RESOLUTION;
    
    //let buf_size = buffer.len();
    let mut x = x_start as f32 - HALF_RES;
    let mut y = y_start as f32 - HALF_RES;
    let scale = zoom / FRES;
    for pixel in buffer.iter_mut() {
            let real = x * scale + x_off;
            let imag = y * scale + y_off;
            let c = color_fn(mandelbrot(real, imag));
            *pixel = c;
            x += 1.0;
            if x >= HALF_RES {
                x = -HALF_RES;
                y += 1.0;
            }
    }
}

// map v to [-r, r]
fn map_to(v : f32, vmin : f32, vmax : f32, range : f32) -> f32 {

    // Get v to 0-1
    let v01 = (v - vmin) / (vmax - vmin);
    (range * 2.0 * v01) - range
}

fn julia(x_start : f32, y_start : f32, angle : f32) -> u32 {
    let mut re = x_start;
    let mut im = y_start;

    let mut iteration = 0;
    while iteration < MAX_ITER {
        let re_new = re*re - im*im;
        let im_new = 2.0 * re * im;
        
        re = re_new + angle.cos();
        im = im_new - angle.sin();
        
        if (re + im).abs() > 2.0 {
            break;
        }
        
        iteration += 1;
    }
    iteration
}

fn julia_chunk(escape_radius : f32, buffer : &mut[u32], offset : usize, angle : f32) {
    let x_start = offset % TEX_RESOLUTION;
    let y_start = offset / TEX_RESOLUTION;
    
    let mut x = x_start as f32;
    let mut y = y_start as f32;
    
    for pix in buffer {
    
        let real = map_to(x, 0.0, FRES-1.0, escape_radius);
        let imag = map_to(y, 0.0, FRES-1.0, escape_radius);

        let c = get_color(julia(real, imag, angle));
        *pix = c;
        
        x += 1.0;
        if x >= FRES {
            x -= FRES;
            y += 1.0;
        }
    }
}

fn transform_vert_chunk(
    model_verts: &[Vertex],
    xformed_verts: &mut [Vertex],
    world_to_vp: &Matrix4,
    base_index: usize,
) {
    for (i, xv) in xformed_verts.iter_mut().enumerate() {
        let j = base_index + i;

        let v4 = transform_point4(&model_verts[j].pos, &world_to_vp);
        xv.pos.x = v4.x / v4.w;
        xv.pos.y = v4.y / v4.w;
        xv.pos.z = v4.z / v4.w;

        xv.uv.x = model_verts[j].uv.x * (FRES-1.0);
        xv.uv.y = model_verts[j].uv.y * (FRES-1.0);
    }
}

fn main() 
{
    let mut window = Window::new("rust_julia - ESC to exit",
                                RESOLUTION, RESOLUTION,
                                WindowOptions::default()).unwrap_or_else(|e| {
        panic!("{}", e);
    });
    
    //let julia_texture = [Arc::new(Mutex::new(vec![0; TEX_RESOLUTION*TEX_RESOLUTION])),
        //Arc::new(Mutex::new(vec![0; TEX_RESOLUTION*TEX_RESOLUTION]))];
    //let thread_tex = [Arc::clone(&julia_texture[0]), Arc::clone(&julia_texture[1])];
    //let mandelbrot_texture = [Arc::new(Mutex::new(vec![0; TEX_RESOLUTION*TEX_RESOLUTION])),
        //Arc::new(Mutex::new(vec![0; TEX_RESOLUTION*TEX_RESOLUTION]))];
    //let mb_thread_tex = [Arc::clone(&mandelbrot_texture[0]), Arc::clone(&mandelbrot_texture[1])];
    
    struct UnsafePtr {
        ptr : *mut u32
    };
    unsafe impl Send for UnsafePtr {};
    let mut julia_texture = [ vec![0; TEX_RESOLUTION*TEX_RESOLUTION], vec![0; TEX_RESOLUTION*TEX_RESOLUTION] ];
    let julia_unsafe_ptrs = [   UnsafePtr { ptr : julia_texture[0].as_mut_ptr() },
                                UnsafePtr { ptr : julia_texture[1].as_mut_ptr() } ];

    let mut mandelbrot_texture = [ vec![0; TEX_RESOLUTION*TEX_RESOLUTION], vec![0; TEX_RESOLUTION*TEX_RESOLUTION] ];
    let mb_unsafe_ptrs = [  UnsafePtr { ptr : mandelbrot_texture[0].as_mut_ptr() },
                            UnsafePtr { ptr : mandelbrot_texture[1].as_mut_ptr() } ];
                               
//    let mut zoom = 8.0;
    let x_off = 0.0;
    let y_off = 0.0;
    
    let cube_verts_world = [
        Vertex { pos : Vector3::new(-0.25, -0.25, -0.25),   uv : Vector2::new(0.0, 0.0) },
        Vertex { pos : Vector3::new(0.25, -0.25, -0.25),    uv : Vector2::new(1.0, 0.0) },
        Vertex { pos : Vector3::new(0.25, 0.25, -0.25),     uv : Vector2::new(1.0, 1.0) },
        Vertex { pos : Vector3::new(-0.25, 0.25, -0.25),    uv : Vector2::new(0.0, 1.0) },

        Vertex { pos : Vector3::new(0.25, -0.25, 0.25),     uv : Vector2::new(0.0, 0.0) },
        Vertex { pos : Vector3::new(-0.25, -0.25, 0.25),    uv : Vector2::new(1.0, 0.0) },
        Vertex { pos : Vector3::new(-0.25, 0.25, 0.25),     uv : Vector2::new(1.0, 1.0) },
        Vertex { pos : Vector3::new(0.25, 0.25, 0.25),      uv : Vector2::new(0.0, 1.0) },
        
        // Diff UV for the top face
        Vertex { pos : Vector3::new(-0.25, -0.25, -0.25),   uv : Vector2::new(1.0, 1.0) },
        Vertex { pos : Vector3::new(0.25, -0.25, -0.25),    uv : Vector2::new(0.0, 1.0) },
        
        //..and bottom
        Vertex { pos : Vector3::new(0.25, 0.25, -0.25),     uv : Vector2::new(0.0, 0.0) },
        Vertex { pos : Vector3::new(-0.25, 0.25, -0.25),    uv : Vector2::new(1.0, 0.0) },
    ];
    let null_v = Vertex {
        pos: ZERO_VECTOR,
        //normal: ZERO_VECTOR,
        uv: ZERO_VECTOR2,
    };
    let mut cube_verts: Vec<Vertex> = vec![null_v; cube_verts_world.len()];
    
    let faces = [
          Triangle { v0 : 0, v1 : 1, v2 : 2 },
          Triangle { v0 : 2, v1 : 3, v2 : 0 },
          
          Triangle { v0 : 4, v1 : 5, v2 : 6 },
          Triangle { v0 : 6, v1 : 7, v2 : 4 },
          
          Triangle { v0 : 1, v1 : 4, v2 : 7 },
          Triangle { v0 : 7, v1 : 2, v2 : 1 },
          
          Triangle { v0 : 5, v1 : 0, v2 : 3 },
          Triangle { v0 : 3, v1 : 6, v2 : 5 },
          
          Triangle { v0 : 5, v1 : 4, v2 : 9 },
          Triangle { v0 : 9, v1 : 8, v2 : 5 },
          
          Triangle { v0 : 7, v1 : 6, v2 : 11 },
          Triangle { v0 : 11, v1 : 10, v2 : 7 }
    ];

    let (tx, rx) = channel();
    let (tx_done, rx_done) = channel();

    let (tx_mb, rx_mb) = channel();
    let (tx_mb_done, rx_mb_done) = channel();
    
    let mut rot_angle = 0.0 as Rfloat;
    
    let julia_thread = thread::spawn(move || {
        let mut angle = 0.0;
        loop
        {
            // Wait until we get something to work with
            let msg = rx.recv().unwrap();
            if msg == 2 {
                break;  // 2 = finish
            }
        
            let start_time = Instant::now();
        
            //let mut tex = thread_tex[msg as usize].lock().unwrap();
            let tex : &mut [u32];
            unsafe { 
                tex = slice::from_raw_parts_mut(julia_unsafe_ptrs[msg as usize].ptr, TEX_RESOLUTION*TEX_RESOLUTION);
            }
            tex.par_chunks_mut(CHUNK_SIZE*CHUNK_SIZE)
                .enumerate()
                .map(|mut x| julia_chunk(2.0,&mut x.1, x.0*CHUNK_SIZE*CHUNK_SIZE, angle))
                .collect::<Vec<_>>();
                
            tx_done.send(1).unwrap();
            let time_taken = Instant::now().duration_since(start_time);
            let time_taken_dbl = time_taken.as_secs() as f64 + time_taken.subsec_nanos() as f64 * 1e-9;
        
            angle += time_taken_dbl as f32 * 2.0;
        }
    });
    
    let mandelbrot_thread = thread::spawn(move || {
        let mut mb_zoom = 0.0;
        let max_zoom = 30.0;
        let min_zoom = -30.0;
        //let mut zoom_dir = 1.0;
        let mut zoom_speed = 20.0;
        loop
        {
            // Wait until we get something to work with
            let msg = rx_mb.recv().unwrap();
            if msg == 2 {
                break;  // 2 = finish
            }
        
            let start_time = Instant::now();
        
//            let mut tex = mb_thread_tex[msg as usize].lock().unwrap();
            let tex : &mut [u32];
            unsafe { 
                tex = slice::from_raw_parts_mut(mb_unsafe_ptrs[msg as usize].ptr, TEX_RESOLUTION*TEX_RESOLUTION);
            }
            
            tex.par_chunks_mut(CHUNK_SIZE*CHUNK_SIZE)
                .enumerate()
                .map(|mut x| mandelbrot_chunk(mb_zoom, x_off, y_off, &get_color2, &mut x.1, x.0*CHUNK_SIZE*CHUNK_SIZE))
                .collect::<Vec<_>>();
                
            tx_mb_done.send(1).unwrap();
            let time_taken = Instant::now().duration_since(start_time);
            let time_taken_dbl = time_taken.as_secs() as f64 + time_taken.subsec_nanos() as f64 * 1e-9;
            
            mb_zoom += time_taken_dbl as f32 * zoom_speed;
            if (mb_zoom > max_zoom && zoom_speed > 0.0) || (mb_zoom < min_zoom && zoom_speed < 0.0) {
                zoom_speed *= -1.0;
            }
        }
    });
    
    let mut job_buffer_idx = 0;
    let mut mb_job_buffer_idx = 0;
    tx.send(job_buffer_idx).unwrap();
    tx_mb.send(mb_job_buffer_idx).unwrap();
    
    let fov = 60.0;
    let proj_matrix = perspective_projection_matrix(fov, 1.0, 0.1, 1000.0);
    let vp_matrix = viewport_matrix(RESOLUTION as Rfloat, RESOLUTION as Rfloat);
    let vpm = view_matrix(4.0) * proj_matrix * vp_matrix;
    let mut world_to_vp : Matrix4;

    let win_min = Point2 { x : 0, y : 0 };
    let win_max = Point2 { x : RESOLUTION as i32, y : RESOLUTION as i32 };
    
    let mut framebuffer: Vec<u32> = vec![0; RESOLUTION * RESOLUTION];
    
    while window.is_open() && !window.is_key_down(Key::Escape)
    {
        framebuffer.clear();
        framebuffer.resize(RESOLUTION * RESOLUTION, 0xFFBB_B7FF);
    
        let start_time = Instant::now();
        
        // See if new buffer is ready
        let done_msg = rx_done.try_recv();
        if done_msg.is_ok() {
            // Flip
            job_buffer_idx ^= 1;
            tx.send(job_buffer_idx).unwrap();
        }
        let mb_done = rx_mb_done.try_recv();
        if mb_done.is_ok() {
            mb_job_buffer_idx ^= 1;
            tx_mb.send(mb_job_buffer_idx).unwrap();
        }
    
        let rot = rotx_matrix(rot_angle) * rotz_matrix(rot_angle) * roty_matrix(rot_angle);
        world_to_vp = rot * vpm;
    
        //framebuffer.par_chunks_mut(CHUNK_SIZE*CHUNK_SIZE)
            //.enumerate()
            //.map(|mut x| julia_chunk(2.0,&mut x.1, x.0*CHUNK_SIZE*CHUNK_SIZE, angle))
            //.collect::<Vec<_>>();
        
        // Single thread
        //julia_chunk(2.0, framebuffer.as_mut_slice(), 0, angle);

        const VERT_CHUNK_SIZE: usize = 512;
        cube_verts
            .par_chunks_mut(VERT_CHUNK_SIZE)
            .enumerate()
            .for_each(|(i, mut chunk)| {
                transform_vert_chunk(&cube_verts_world, &mut chunk, &world_to_vp, i * VERT_CHUNK_SIZE);
            });
        
        let buf_tex = &julia_texture[job_buffer_idx ^ 1];//.lock().unwrap();
        let mb_tex = &mandelbrot_texture[mb_job_buffer_idx ^ 1];//.lock().unwrap();
        
        for (ti, t) in faces.iter().enumerate()
        {
            let tri_tex = if (ti / 6) & 0x1 != 0 { &buf_tex } else { &mb_tex }; 
            let v0 = &cube_verts[t.v0];
            let v1 = &cube_verts[t.v1];
            let v2 = &cube_verts[t.v2];
            draw_triangle(
                v0,
                v1,
                v2,
                &win_min, &win_max,
                framebuffer.as_mut_slice(),
                tri_tex.as_slice(),
            );
        }
        
        let time_taken = Instant::now().duration_since(start_time);
        let time_taken_dbl = time_taken.as_secs() as f64 + time_taken.subsec_nanos() as f64 * 1e-9;
        let fps = (1.0 / time_taken_dbl) as u32;
        window.set_title(&format!("FPS {}", fps));
        
        window.update_with_buffer(framebuffer.as_slice(), RESOLUTION, RESOLUTION).unwrap();
        rot_angle += time_taken_dbl as f32 * 70.0;
    }
    
    tx.send(2).unwrap();
    julia_thread.join().unwrap();
    
    tx_mb.send(2).unwrap();
    mandelbrot_thread.join().unwrap();
}
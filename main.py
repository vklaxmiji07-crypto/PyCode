import sys
import os
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QFileDialog, QLabel, QHBoxLayout, QFrame, QCheckBox,
                             QLineEdit, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (QSurfaceFormat, QVector3D, QMatrix4x4, QImage, QOpenGLTexture,
                         QDoubleValidator)
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import (QOpenGLVertexArrayObject, QOpenGLBuffer,
                         QOpenGLShader, QOpenGLShaderProgram)
from OpenGL.GL import *

# --- PBR Shaders (Unchanged) ---
PBR_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;

out VS_OUT {
    vec3 FragPos;
    vec2 TexCoords;
    mat3 TBN;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    vs_out.FragPos = vec3(model * vec4(aPos, 1.0));
    vs_out.TexCoords = aTexCoords;

    mat3 normalMatrix = transpose(inverse(mat3(model)));
    vec3 T = normalize(normalMatrix * aTangent);
    vec3 B = normalize(normalMatrix * aBitangent);
    vec3 N = normalize(normalMatrix * aNormal);
    vs_out.TBN = mat3(T, B, N);

    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

PBR_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec2 TexCoords;
    mat3 TBN;
} fs_in;

uniform sampler2D albedoMap;
uniform sampler2D normalMap;

uniform vec3 defaultAlbedo;
uniform float defaultMetallic;
uniform float defaultRoughness;
uniform float defaultAo;

uniform bool useAlbedoMap;
uniform bool useNormalMap;

#define MAX_LIGHTS 4
uniform int lightCount;
uniform vec3 lightPositions[MAX_LIGHTS];
uniform vec3 lightColors[MAX_LIGHTS];
uniform vec3 camPos;

const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float r) {
    float a=r*r, a2=a*a, NdotH=max(dot(N,H),0.0), NdotH2=NdotH*NdotH;
    float nom=a2, denom=(NdotH2*(a2-1.0)+1.0);
    return nom/(PI*denom*denom);
}

float GeometrySchlickGGX(float NdotV, float r) {
    float k=(r+1.0)*(r+1.0)/8.0;
    return NdotV/(NdotV*(1.0-k)+k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float r) {
    return GeometrySchlickGGX(max(dot(N,V),0.0),r) * GeometrySchlickGGX(max(dot(N,L),0.0),r);
}

void main()
{
    vec3 albedo = useAlbedoMap ? texture(albedoMap, fs_in.TexCoords).rgb : defaultAlbedo;
    float metallic = defaultMetallic;
    float roughness = defaultRoughness;
    float ao = defaultAo;

    vec3 N = useNormalMap ? normalize(fs_in.TBN * (texture(normalMap, fs_in.TexCoords).rgb * 2.0 - 1.0)) : normalize(fs_in.TBN[2]);
    vec3 V = normalize(camPos - fs_in.FragPos);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 Lo = vec3(0.0);
    for(int i = 0; i < lightCount; ++i)
    {
        vec3 L = normalize(lightPositions[i] - fs_in.FragPos);
        vec3 H = normalize(V + L);
        float distance = length(lightPositions[i] - fs_in.FragPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;

        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 kS = F;
        vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + (NDF * G * F) / (4.0*max(dot(N,V),0.0)*max(dot(N,L),0.0)+0.0001)) * radiance * NdotL;
    }

    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo;

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    FragColor = vec4(color, 1.0);
}
"""

# --- Depth Pre-Pass Shaders ---
GIZMO_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

GIZMO_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

uniform vec3 color;

void main() {
    FragColor = vec4(color, 1.0);
}
"""
DEPTH_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

DEPTH_FRAGMENT_SHADER = """
#version 330 core
void main() {
    // No color output needed - depth only
}
"""


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pbr_program = None
        self.depth_program = None
        self.gizmo_program = None
        self.vao, self.vbo, self.ebo = None, None, None
        self.floor_vao, self.floor_vbo, self.floor_ebo = None, None, None
        self.gizmo_vao, self.gizmo_vbo = None, None
        self.tex_albedo, self.tex_normal = None, None

        self.vertices, self.indices, self.model_loaded = None, None, False
        self.rotation_x, self.rotation_y, self.zoom = 0.0, 0.0, -5.0
        self.x_pan, self.y_pan, self.model_scale = 0.0, 0.0, 1.0
        self.model_position = QVector3D(0, 0, 0)

        self.lights = [
            {'pos': QVector3D(5.0, 5.0, 5.0), 'color': QVector3D(300.0, 300.0, 300.0), 'enabled': True},
            {'pos': QVector3D(-5.0, 5.0, 5.0), 'color': QVector3D(300.0, 0.0, 0.0), 'enabled': False},
            {'pos': QVector3D(5.0, -5.0, 5.0), 'color': QVector3D(0.0, 300.0, 0.0), 'enabled': False},
            {'pos': QVector3D(-5.0, -5.0, 5.0), 'color': QVector3D(0.0, 0.0, 300.0), 'enabled': False}
        ]

        self.last_pos = None
        self.model_matrix = QMatrix4x4()
        self.floor_model_matrix = QMatrix4x4()
        self.view_matrix = QMatrix4x4()
        self.projection_matrix = QMatrix4x4()

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.parent_window = parent
        self.setFocusPolicy(Qt.StrongFocus)
        self.depth_prepass_enabled = True

        # Gizmo interaction state
        self.selected_axis = -1 # -1: none, 0: X, 1: Y, 2: Z
        self.gizmo_interaction_active = False
        self.last_mouse_pos_on_drag_start = None

    def _compile_shader(self, v_src, f_src, name="Shader"):
        shader_prog = QOpenGLShaderProgram(self)
        if not shader_prog.addShaderFromSourceCode(QOpenGLShader.Vertex, v_src):
            print(f"ERROR: {name} Vertex shader compilation failed:", shader_prog.log())
            return None
        if not shader_prog.addShaderFromSourceCode(QOpenGLShader.Fragment, f_src):
            print(f"ERROR: {name} Fragment shader compilation failed:", shader_prog.log())
            return None
        if not shader_prog.link():
            print(f"ERROR: {name} Shader program linking failed:", shader_prog.log())
            return None
        print(f"✓ {name} shader compiled and linked successfully.")
        return shader_prog

    def load_texture(self, filename, texture_type):
        self.makeCurrent()
        image = QImage(filename)
        if image.isNull():
            print(f"Failed to load texture: {filename}")
            return

        if texture_type == 'albedo':
            if self.tex_albedo:
                self.tex_albedo.destroy()
            self.tex_albedo = QOpenGLTexture(image.mirrored())
            self.tex_albedo.setMinificationFilter(QOpenGLTexture.LinearMipMapLinear)
            self.tex_albedo.setMagnificationFilter(QOpenGLTexture.Linear)
            self.tex_albedo.setWrapMode(QOpenGLTexture.Repeat)
        elif texture_type == 'normal':
            if self.tex_normal:
                self.tex_normal.destroy()
            self.tex_normal = QOpenGLTexture(image.mirrored())
            self.tex_normal.setMinificationFilter(QOpenGLTexture.LinearMipMapLinear)
            self.tex_normal.setMagnificationFilter(QOpenGLTexture.Linear)
            self.tex_normal.setWrapMode(QOpenGLTexture.Repeat)

        self.update()
        print(f"✓ Loaded {texture_type} map: {os.path.basename(filename)}")

    def load_obj_file(self, filename):
        print(f"Loading OBJ file: {filename}")
        v, vt, vn = [], [], []
        vert_data_map = {}
        final_vertices, indices = [], []

        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    v.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'vt':
                    vt.append([float(x) for x in parts[1:3]])
                elif parts[0] == 'vn':
                    vn.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'f':
                    face_verts_str = parts[1:]
                    # --- FIX: Robust fan triangulation for polygons to prevent IndexError ---
                    if len(face_verts_str) >= 3:
                        for i in range(len(face_verts_str) - 2):
                            # Creates triangles (v0, v1, v2), (v0, v2, v3), etc.
                            tri_indices = [0, i + 1, i + 2]
                            for j in range(3):
                                vert_str = face_verts_str[tri_indices[j]]
                                if vert_str not in vert_data_map:
                                    vert_data_map[vert_str] = len(final_vertices)
                                    # Safely parse vertex indices
                                    try:
                                        v_idx_str, vt_idx_str, vn_idx_str = (vert_str.split('/') + ['', ''])[:3]
                                        v_idx = int(v_idx_str) - 1

                                        pos = v[v_idx] if v_idx < len(v) else [0,0,0]

                                        vt_idx = int(vt_idx_str) - 1 if vt_idx_str else -1
                                        tex = vt[vt_idx] if vt_idx != -1 and vt_idx < len(vt) else [0,0]

                                        vn_idx = int(vn_idx_str) - 1 if vn_idx_str else -1
                                        nrm = vn[vn_idx] if vn_idx != -1 and vn_idx < len(vn) else [0,1,0]

                                        final_vertices.append([pos, nrm, tex, [0,0,0], [0,0,0]])
                                    except (ValueError, IndexError) as e:
                                        print(f"Warning: Skipping malformed face vertex data '{vert_str}'. Error: {e}")
                                        # Add a placeholder to avoid breaking index map
                                        if vert_str not in vert_data_map:
                                            vert_data_map[vert_str] = len(final_vertices)
                                            final_vertices.append([[0,0,0], [0,1,0], [0,0], [0,0,0], [0,0,0]])

                                indices.append(vert_data_map[vert_str])

        if not final_vertices:
            print("ERROR: No vertices found in OBJ file")
            return False

        # Convert to numpy array
        vertices_np = np.zeros((len(final_vertices), 14), dtype=np.float32)
        for i, vert in enumerate(final_vertices):
            vertices_np[i, 0:3] = vert[0]   # position
            vertices_np[i, 3:6] = vert[1]   # normal
            vertices_np[i, 6:8] = vert[2]   # uv
            vertices_np[i, 8:11] = vert[3]  # tangent (will be calculated)
            vertices_np[i, 11:14] = vert[4] # bitangent (will be calculated)

        # Calculate tangents and bitangents
        for i in range(0, len(indices), 3):
            i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
            pos0, pos1, pos2 = vertices_np[i0, 0:3], vertices_np[i1, 0:3], vertices_np[i2, 0:3]
            uv0, uv1, uv2 = vertices_np[i0, 6:8], vertices_np[i1, 6:8], vertices_np[i2, 6:8]

            edge1, edge2 = pos1 - pos0, pos2 - pos0
            deltaUV1, deltaUV2 = uv1 - uv0, uv2 - uv0

            denom = (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])
            f = 1.0 / denom if denom != 0 else 0.0

            tangent = f * (deltaUV2[1] * edge1 - deltaUV1[1] * edge2)
            bitangent = f * (-deltaUV2[0] * edge1 + deltaUV1[0] * edge2)

            # Accumulate for averaging
            for idx in [i0, i1, i2]:
                vertices_np[idx, 8:11] += tangent
                vertices_np[idx, 11:14] += bitangent

        # Normalize tangents and bitangents
        for i in range(len(vertices_np)):
            # Gram-Schmidt orthogonalize
            t = vertices_np[i, 8:11]
            n = vertices_np[i, 3:6]
            t_norm = np.linalg.norm(t)
            if t_norm > 0:
                t = t / t_norm
                # Make tangent orthogonal to normal
                t = t - n * np.dot(n, t)
                t_norm = np.linalg.norm(t)
                if t_norm > 0:
                    vertices_np[i, 8:11] = t / t_norm

        self.vertices = vertices_np
        self.indices = np.array(indices, dtype=np.uint32)

        max_coord = np.max(np.abs(self.vertices[:, 0:3]))
        self.model_scale = 2.0 / max_coord if max_coord > 0 else 1.0

        self.makeCurrent()
        self._create_gpu_buffers()
        self.model_loaded = True
        self.update()

        print(f"✓ Model loaded: {len(self.vertices)} vertices, {len(self.indices)//3} triangles")
        return True

    def _create_gpu_buffers(self):
        if self.vao:
            self.vao.destroy()
        if self.vbo:
            self.vbo.destroy()
        if self.ebo:
            self.ebo.destroy()

        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo.create()
        self.vbo.bind()
        self.vbo.allocate(self.vertices.tobytes(), self.vertices.nbytes)

        stride = 14 * 4  # 14 floats, 4 bytes each
        # Layout: position(3), normal(3), uv(2), tangent(3), bitangent(3)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8 * 4))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(11 * 4))
        glEnableVertexAttribArray(4)

        self.ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        self.ebo.create()
        self.ebo.bind()
        self.ebo.allocate(self.indices.tobytes(), self.indices.nbytes)

        self.vao.release()

    def _create_floor_buffers(self):
        # Floor vertices (pos, normal, uv, tangent, bitangent)
        floor_vertices = np.array([
            # Positions            # Normals      # TexCoords  # Tangent       # Bitangent
            -10.0, -1.5,  10.0,  0.0, 1.0, 0.0,   0.0, 10.0,  1.0, 0.0, 0.0,  0.0, 0.0, -1.0,
             10.0, -1.5,  10.0,  0.0, 1.0, 0.0,  10.0, 10.0,  1.0, 0.0, 0.0,  0.0, 0.0, -1.0,
             10.0, -1.5, -10.0,  0.0, 1.0, 0.0,  10.0,  0.0,  1.0, 0.0, 0.0,  0.0, 0.0, -1.0,
            -10.0, -1.5, -10.0,  0.0, 1.0, 0.0,   0.0,  0.0,  1.0, 0.0, 0.0,  0.0, 0.0, -1.0,
        ], dtype=np.float32)

        floor_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self.floor_vao = QOpenGLVertexArrayObject()
        self.floor_vao.create()
        self.floor_vao.bind()

        self.floor_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.floor_vbo.create()
        self.floor_vbo.bind()
        self.floor_vbo.allocate(floor_vertices.tobytes(), floor_vertices.nbytes)

        stride = 14 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8 * 4))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(11 * 4))
        glEnableVertexAttribArray(4)

        self.floor_ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        self.floor_ebo.create()
        self.floor_ebo.bind()
        self.floor_ebo.allocate(floor_indices.tobytes(), floor_indices.nbytes)

        self.floor_vao.release()
        print("✓ Floor buffers created")

    def _create_gizmo_buffers(self):
        # Simple gizmo: 3 lines for axes
        gizmo_vertices = np.array([
            # X-axis (red)
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            # Y-axis (green)
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            # Z-axis (blue)
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
        ], dtype=np.float32)

        self.gizmo_vao = QOpenGLVertexArrayObject()
        self.gizmo_vao.create()
        self.gizmo_vao.bind()

        self.gizmo_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.gizmo_vbo.create()
        self.gizmo_vbo.bind()
        self.gizmo_vbo.allocate(gizmo_vertices.tobytes(), gizmo_vertices.nbytes)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        self.gizmo_vao.release()
        print("✓ Gizmo buffers created")

    def initializeGL(self):
        print("Initializing OpenGL...")
        self.pbr_program = self._compile_shader(PBR_VERTEX_SHADER, PBR_FRAGMENT_SHADER, "PBR")
        self.depth_program = self._compile_shader(DEPTH_VERTEX_SHADER, DEPTH_FRAGMENT_SHADER, "Depth")
        self.gizmo_program = self._compile_shader(GIZMO_VERTEX_SHADER, GIZMO_FRAGMENT_SHADER, "Gizmo")

        if not self.pbr_program or not self.depth_program or not self.gizmo_program:
            print("FATAL: Failed to compile shaders!")
            return

        self._create_floor_buffers()
        self._create_gizmo_buffers()

        glClearColor(0.1, 0.1, 0.15, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        print("✓ OpenGL initialized")

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        self.projection_matrix.setToIdentity()
        if h > 0:
            self.projection_matrix.perspective(45.0, w / h, 0.1, 100.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.view_matrix.setToIdentity()
        self.view_matrix.translate(self.x_pan, self.y_pan, self.zoom)
        self.view_matrix.rotate(self.rotation_x, 1, 0, 0)
        self.view_matrix.rotate(self.rotation_y, 0, 1, 0)

        # --- Render Floor ---
        self.floor_model_matrix.setToIdentity()

        self.pbr_program.bind()
        self.pbr_program.setUniformValue("projection", self.projection_matrix)
        self.pbr_program.setUniformValue("view", self.view_matrix)
        self.pbr_program.setUniformValue("model", self.floor_model_matrix)
        self.pbr_program.setUniformValue("camPos", QVector3D(0,0,5))

        enabled_lights = [l for l in self.lights if l['enabled']]
        self.pbr_program.setUniformValue("lightCount", len(enabled_lights))
        for i, light in enumerate(enabled_lights):
            self.pbr_program.setUniformValue(f"lightPositions[{i}]", light['pos'])
            self.pbr_program.setUniformValue(f"lightColors[{i}]", light['color'])

        self.pbr_program.setUniformValue("useAlbedoMap", False)
        self.pbr_program.setUniformValue("defaultAlbedo", QVector3D(0.4, 0.4, 0.4))
        self.pbr_program.setUniformValue("useNormalMap", False)
        self.pbr_program.setUniformValue("defaultMetallic", 0.1)
        self.pbr_program.setUniformValue("defaultRoughness", 0.8)
        self.pbr_program.setUniformValue("defaultAo", 1.0)

        self.floor_vao.bind()
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        self.floor_vao.release()

        if not self.model_loaded or not self.pbr_program or not self.depth_program:
            return

        self.model_matrix.setToIdentity()
        self.model_matrix.translate(self.model_position)
        self.model_matrix.scale(self.model_scale)

        self.vao.bind()

        if self.depth_prepass_enabled:
            # --- OPTIMIZATION: Depth Pre-Pass ---
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
            glDepthMask(GL_TRUE)

            self.depth_program.bind()
            self.depth_program.setUniformValue("projection", self.projection_matrix)
            self.depth_program.setUniformValue("view", self.view_matrix)
            self.depth_program.setUniformValue("model", self.model_matrix)
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
            self.depth_program.release()

            # --- Main PBR Rendering Pass ---
            glDepthFunc(GL_LEQUAL)
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
            glDepthMask(GL_FALSE)
        else:
            # --- Standard Forward Rendering ---
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
            glDepthMask(GL_TRUE)

        self.pbr_program.bind()

        self.pbr_program.setUniformValue("projection", self.projection_matrix)
        self.pbr_program.setUniformValue("view", self.view_matrix)
        self.pbr_program.setUniformValue("model", self.model_matrix)

        enabled_lights = [l for l in self.lights if l['enabled']]
        self.pbr_program.setUniformValue("lightCount", len(enabled_lights))
        for i, light in enumerate(enabled_lights):
            self.pbr_program.setUniformValue(f"lightPositions[{i}]", light['pos'])
            self.pbr_program.setUniformValue(f"lightColors[{i}]", light['color'])

        self.pbr_program.setUniformValue("useAlbedoMap", bool(self.tex_albedo))
        if self.tex_albedo:
            glActiveTexture(GL_TEXTURE0)
            self.tex_albedo.bind()
            self.pbr_program.setUniformValue("albedoMap", 0)
        else:
            self.pbr_program.setUniformValue("defaultAlbedo", QVector3D(0.5, 0.5, 0.5))

        self.pbr_program.setUniformValue("useNormalMap", bool(self.tex_normal))
        if self.tex_normal:
            glActiveTexture(GL_TEXTURE1)
            self.tex_normal.bind()
            self.pbr_program.setUniformValue("normalMap", 1)

        self.pbr_program.setUniformValue("defaultMetallic", 0.9)
        self.pbr_program.setUniformValue("defaultRoughness", 0.2)
        self.pbr_program.setUniformValue("defaultAo", 1.0)

        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

        self.pbr_program.release()
        self.vao.release()

        # --- Render Gizmo ---
        glDisable(GL_DEPTH_TEST)
        self.gizmo_program.bind()
        self.gizmo_program.setUniformValue("projection", self.projection_matrix)
        self.gizmo_program.setUniformValue("view", self.view_matrix)

        gizmo_model_matrix = QMatrix4x4()
        gizmo_model_matrix.translate(self.model_position)
        # Scale gizmo to be visible regardless of zoom
        gizmo_scale = abs(self.zoom) * 0.1
        gizmo_model_matrix.scale(gizmo_scale)

        self.gizmo_program.setUniformValue("model", gizmo_model_matrix)

        self.gizmo_vao.bind()
        # Draw X axis
        self.gizmo_program.setUniformValue("color", QVector3D(1.0, 0.0, 0.0))
        glDrawArrays(GL_LINES, 0, 2)
        # Draw Y axis
        self.gizmo_program.setUniformValue("color", QVector3D(0.0, 1.0, 0.0))
        glDrawArrays(GL_LINES, 2, 2)
        # Draw Z axis
        self.gizmo_program.setUniformValue("color", QVector3D(0.0, 0.0, 1.0))
        glDrawArrays(GL_LINES, 4, 2)
        self.gizmo_vao.release()
        self.gizmo_program.release()

        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)


        # FPS Calculation
        self.frame_count += 1
        current_time = time.time()
        if (current_time - self.last_fps_time) >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            if self.parent_window:
                self.parent_window.update_fps(fps)
            self.frame_count = 0
            self.last_fps_time = current_time

    def wheelEvent(self, e):
        self.zoom += e.angleDelta().y() / 120.0 * 0.3
        self.zoom = np.clip(self.zoom, -50.0, -0.5)
        self.update()

    def mousePressEvent(self, e):
        self.last_pos = e.pos()

        # Check for gizmo intersection
        self._check_gizmo_intersection(e.pos())

    def _check_gizmo_intersection(self, pos):
        # Unproject mouse coordinates to a 3D ray
        near_point = QVector3D(pos.x(), self.height() - pos.y(), 0.0)
        far_point = QVector3D(pos.x(), self.height() - pos.y(), 1.0)

        view_projection_matrix = self.projection_matrix * self.view_matrix
        inv_vp = view_projection_matrix.inverted()[0]

        near_world = inv_vp.map(near_point)
        far_world = inv_vp.map(far_point)

        ray_dir = (far_world - near_world).normalized()
        ray_origin = near_world

        # --- Bounding Box check for each axis ---
        gizmo_scale = abs(self.zoom) * 0.1
        axis_length = 1.0 * gizmo_scale
        axis_thickness = 0.1 * gizmo_scale

        # Bounding boxes in local space
        bboxes = [
            (QVector3D(0, -axis_thickness, -axis_thickness), QVector3D(axis_length, axis_thickness, axis_thickness)), # X
            (QVector3D(-axis_thickness, 0, -axis_thickness), QVector3D(axis_thickness, axis_length, axis_thickness)), # Y
            (QVector3D(-axis_thickness, -axis_thickness, 0), QVector3D(axis_thickness, axis_thickness, axis_length))  # Z
        ]

        self.selected_axis = -1
        min_dist = float('inf')

        for i in range(3):
            min_vec = bboxes[i][0] + self.model_position
            max_vec = bboxes[i][1] + self.model_position

            tmin = 0.0
            tmax = float('inf')

            for j in range(3):
                if abs(ray_dir[j]) < 1e-6:
                    if ray_origin[j] < min_vec[j] or ray_origin[j] > max_vec[j]:
                        tmin = float('inf')
                        tmax = -float('inf')
                        break
                else:
                    t1 = (min_vec[j] - ray_origin[j]) / ray_dir[j]
                    t2 = (max_vec[j] - ray_origin[j]) / ray_dir[j]

                    if t1 > t2: t1, t2 = t2, t1
                    tmin = max(tmin, t1)
                    tmax = min(tmax, t2)

            if tmin <= tmax and tmin < min_dist:
                min_dist = tmin
                self.selected_axis = i

        if self.selected_axis != -1:
            self.gizmo_interaction_active = True
            self.last_mouse_pos_on_drag_start = pos
        else:
            self.gizmo_interaction_active = False


    def mouseMoveEvent(self, e):
        if not self.last_pos:
            return

        if self.gizmo_interaction_active and self.selected_axis != -1:
            # Move object along selected axis
            dx = e.x() - self.last_pos.x()
            dy = e.y() - self.last_pos.y()

            # Determine movement direction based on view
            right_vec = self.view_matrix.inverted()[0].column(0).toVector3D().normalized()
            up_vec = self.view_matrix.inverted()[0].column(1).toVector3D().normalized()

            move_vec = QVector3D()
            if self.selected_axis == 0: # X-axis
                move_vec = QVector3D(1, 0, 0)
            elif self.selected_axis == 1: # Y-axis
                move_vec = QVector3D(0, 1, 0)
            elif self.selected_axis == 2: # Z-axis
                move_vec = QVector3D(0, 0, 1)

            # Project mouse movement onto the axis in screen space
            axis_screen = self.projection_matrix * self.view_matrix * move_vec
            mouse_move_screen = QVector3D(dx, -dy, 0)

            # Determine amount of movement
            # A more robust solution would use plane intersection
            move_speed = abs(self.zoom) * 0.005
            amount = QVector3D.dotProduct(axis_screen.normalized(), mouse_move_screen.normalized()) * mouse_move_screen.length() * move_speed

            self.model_position += move_vec * amount

        elif e.buttons() & Qt.LeftButton:
            dx, dy = e.x() - self.last_pos.x(), e.y() - self.last_pos.y()
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
        elif e.buttons() & Qt.MiddleButton:
            dx, dy = e.x() - self.last_pos.x(), e.y() - self.last_pos.y()
            pan_speed = abs(self.zoom) * 0.001
            self.x_pan += dx * pan_speed
            self.y_pan -= dy * pan_speed

        self.last_pos = e.pos()
        self.update()

    def mouseReleaseEvent(self, e):
        self.last_pos = None
        self.gizmo_interaction_active = False
        self.selected_axis = -1
        self.parent_window.update_transform_inputs()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_R:
            self.rotation_x = self.rotation_y = 0.0
            self.x_pan = self.y_pan = 0.0
            self.zoom = -5.0
            self.update()
        elif e.key() in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4]:
            index = e.key() - Qt.Key_1
            if index < len(self.lights):
                self.lights[index]['enabled'] = not self.lights[index]['enabled']
                print(f"Light {index+1} {'enabled' if self.lights[index]['enabled'] else 'disabled'}")
                self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimized PBR 3D Viewer")
        self.setGeometry(100, 100, 1280, 800)

        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setSamples(4)
        fmt.setSwapInterval(1)
        QSurfaceFormat.setDefaultFormat(fmt)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left control panel
        control_panel = QWidget()
        control_panel.setFixedWidth(260)
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)

        # Right OpenGL view
        self.glWidget = GLWidget(self)
        main_layout.addWidget(self.glWidget, 1)

        # --- ADDED: Optimization toggle ---
        self.depth_prepass_checkbox = QCheckBox("Enable Depth Pre-Pass")
        self.depth_prepass_checkbox.setChecked(True)
        self.depth_prepass_checkbox.toggled.connect(self.toggle_depth_prepass)
        control_layout.addWidget(self.depth_prepass_checkbox)


        # Styling
        self.setStyleSheet("""
            QPushButton {
                background-color: #2962FF;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #0039CB;
            }
            QLabel {
                font-size: 12px;
                color: #CFD8DC;
            }
            QFrame {
                border: 1px solid #37474F;
                border-radius: 5px;
                padding: 10px;
                margin: 5px 0;
            }
        """)

        # Model section
        model_frame = QFrame()
        model_layout = QVBoxLayout(model_frame)
        model_layout.addWidget(QLabel("1. Load Model"))
        btn_load_model = QPushButton("📁 Load .obj File")
        btn_load_model.clicked.connect(self.load_model)
        model_layout.addWidget(btn_load_model)
        control_layout.addWidget(model_frame)

        # Texture section
        tex_frame = QFrame()
        tex_layout = QVBoxLayout(tex_frame)
        tex_layout.addWidget(QLabel("2. Load Textures (Optional)"))
        btn_load_albedo = QPushButton("🎨 Load Albedo/Color")
        btn_load_albedo.clicked.connect(lambda: self.load_texture('albedo'))
        tex_layout.addWidget(btn_load_albedo)
        btn_load_normal = QPushButton("📈 Load Normal Map")
        btn_load_normal.clicked.connect(lambda: self.load_texture('normal'))
        tex_layout.addWidget(btn_load_normal)
        control_layout.addWidget(tex_frame)

        # Info section
        info_frame = QFrame()
        info_layout = QVBoxLayout(info_frame)
        info_layout.addWidget(QLabel("Controls:"))
        info_layout.addWidget(QLabel("• Left Mouse: Rotate"))
        info_layout.addWidget(QLabel("• Middle Mouse: Pan"))
        info_layout.addWidget(QLabel("• Scroll: Zoom"))
        info_layout.addWidget(QLabel("• R: Reset View"))
        info_layout.addWidget(QLabel("• 1-4: Toggle Lights"))
        control_layout.addWidget(info_frame)

        # Transform section
        transform_frame = QFrame()
        transform_layout = QGridLayout(transform_frame)
        transform_layout.addWidget(QLabel("Model Position"), 0, 0, 1, 2)

        validator = QDoubleValidator(-1000.0, 1000.0, 3, self)

        self.pos_x_edit = QLineEdit("0.0")
        self.pos_x_edit.setValidator(validator)
        self.pos_x_edit.editingFinished.connect(self.update_model_position_from_inputs)
        transform_layout.addWidget(QLabel("X"), 1, 0)
        transform_layout.addWidget(self.pos_x_edit, 1, 1)

        self.pos_y_edit = QLineEdit("0.0")
        self.pos_y_edit.setValidator(validator)
        self.pos_y_edit.editingFinished.connect(self.update_model_position_from_inputs)
        transform_layout.addWidget(QLabel("Y"), 2, 0)
        transform_layout.addWidget(self.pos_y_edit, 2, 1)

        self.pos_z_edit = QLineEdit("0.0")
        self.pos_z_edit.setValidator(validator)
        self.pos_z_edit.editingFinished.connect(self.update_model_position_from_inputs)
        transform_layout.addWidget(QLabel("Z"), 3, 0)
        transform_layout.addWidget(self.pos_z_edit, 3, 1)

        control_layout.addWidget(transform_frame)


        control_layout.addStretch()

        self.status_label = QLabel("Load a .obj model to begin.")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        control_layout.addWidget(self.status_label)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-weight: bold; color: #4CAF50; padding: 5px;")
        control_layout.addWidget(self.fps_label)

        # Start continuous rendering
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.glWidget.update)
        self.render_timer.start(16)  # ~60 FPS

    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open .obj File", "", "OBJ Files (*.obj);;All Files (*)")

        if filename:
            self.status_label.setText("⏳ Loading model...")
            QApplication.processEvents()

            if self.glWidget.load_obj_file(filename):
                vertex_count = len(self.glWidget.vertices)
                tri_count = len(self.glWidget.indices) // 3
                self.status_label.setText(
                    f"✓ Model Loaded\n{vertex_count:,} vertices\n{tri_count:,} triangles")
            else:
                self.status_label.setText("❌ Error loading model")

    def load_texture(self, tex_type):
        if not self.glWidget.model_loaded:
            self.status_label.setText("⚠ Load a model first!")
            return

        filename, _ = QFileDialog.getOpenFileName(
            self, f"Open {tex_type.capitalize()} Map", "",
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)")

        if filename:
            self.glWidget.load_texture(filename, tex_type)
            self.status_label.setText(f"✓ {tex_type.capitalize()} map loaded")

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def toggle_depth_prepass(self, checked):
        self.glWidget.depth_prepass_enabled = checked
        print(f"Depth Pre-Pass {'Enabled' if checked else 'Disabled'}")

    def update_transform_inputs(self):
        pos = self.glWidget.model_position
        self.pos_x_edit.setText(f"{pos.x():.3f}")
        self.pos_y_edit.setText(f"{pos.y():.3f}")
        self.pos_z_edit.setText(f"{pos.z():.3f}")

    def update_model_position_from_inputs(self):
        x = float(self.pos_x_edit.text())
        y = float(self.pos_y_edit.text())
        z = float(self.pos_z_edit.text())
        self.glWidget.model_position = QVector3D(x, y, z)
        self.glWidget.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

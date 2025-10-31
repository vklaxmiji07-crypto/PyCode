# PyCode
A 3d Python viewer.
import sys
import os
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QFileDialog, QLabel, QHBoxLayout, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QSurfaceFormat, QVector3D, QMatrix4x4, QImage, QOpenGLTexture
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

uniform vec3 lightPositions[1];
uniform vec3 lightColors[1];
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
    for(int i = 0; i < 1; ++i)
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
        self.vao, self.vbo, self.ebo = None, None, None
        self.tex_albedo, self.tex_normal = None, None

        self.vertices, self.indices, self.model_loaded = None, None, False
        self.rotation_x, self.rotation_y, self.zoom = 0.0, 0.0, -5.0
        self.x_pan, self.y_pan, self.model_scale = 0.0, 0.0, 1.0
        
        self.light_pos = QVector3D(5.0, 5.0, 5.0)
        self.light_color = QVector3D(300.0, 300.0, 300.0)
        
        self.last_pos = None
        self.model_matrix = QMatrix4x4()
        self.view_matrix = QMatrix4x4()
        self.projection_matrix = QMatrix4x4()
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.parent_window = parent
        self.setFocusPolicy(Qt.StrongFocus)

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

    def initializeGL(self):
        print("Initializing OpenGL...")
        self.pbr_program = self._compile_shader(PBR_VERTEX_SHADER, PBR_FRAGMENT_SHADER, "PBR")
        self.depth_program = self._compile_shader(DEPTH_VERTEX_SHADER, DEPTH_FRAGMENT_SHADER, "Depth")
        
        if not self.pbr_program or not self.depth_program:
            print("FATAL: Failed to compile shaders!")
            return
        
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
        
        if not self.model_loaded or not self.pbr_program or not self.depth_program:
            return
        
        # Setup model matrix
        self.model_matrix.setToIdentity()
        self.model_matrix.translate(self.x_pan, self.y_pan, self.zoom)
        self.model_matrix.rotate(self.rotation_x, 1, 0, 0)
        self.model_matrix.rotate(self.rotation_y, 0, 1, 0)
        self.model_matrix.scale(self.model_scale)
        
        self.view_matrix.setToIdentity()
        
        self.vao.bind()
        
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
        
        self.pbr_program.bind()
        
        # Set uniforms
        self.pbr_program.setUniformValue("projection", self.projection_matrix)
        self.pbr_program.setUniformValue("view", self.view_matrix)
        self.pbr_program.setUniformValue("model", self.model_matrix)
        self.pbr_program.setUniformValue("camPos", QVector3D(0, 0, 5))
        
        # --- FIX: Set light uniform arrays correctly ---
        self.pbr_program.setUniformValue("lightPositions[0]", self.light_pos)
        self.pbr_program.setUniformValue("lightColors[0]", self.light_color)

        # Texture handling
        self.pbr_program.setUniformValue("useAlbedoMap", bool(self.tex_albedo))
        if self.tex_albedo:
            glActiveTexture(GL_TEXTURE0)
            self.tex_albedo.bind()
            self.pbr_program.setUniformValue("albedoMap", 0)
        else:
            # Default shiny grey material
            self.pbr_program.setUniformValue("defaultAlbedo", QVector3D(0.5, 0.5, 0.5))
        
        self.pbr_program.setUniformValue("useNormalMap", bool(self.tex_normal))
        if self.tex_normal:
            glActiveTexture(GL_TEXTURE1)
            self.tex_normal.bind()
            self.pbr_program.setUniformValue("normalMap", 1)

        # Material properties (shiny grey when no textures)
        self.pbr_program.setUniformValue("defaultMetallic", 0.9)
        self.pbr_program.setUniformValue("defaultRoughness", 0.2)
        self.pbr_program.setUniformValue("defaultAo", 1.0)
        
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        
        self.pbr_program.release()
        self.vao.release()
        
        # Re-enable depth writes for next frame
        glDepthMask(GL_TRUE)

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

    def mouseMoveEvent(self, e):
        if not self.last_pos:
            return
        dx, dy = e.x() - self.last_pos.x(), e.y() - self.last_pos.y()
        if e.buttons() & Qt.LeftButton:
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
        elif e.buttons() & Qt.MiddleButton:
            pan_speed = abs(self.zoom) * 0.001
            self.x_pan += dx * pan_speed
            self.y_pan -= dy * pan_speed
        self.last_pos = e.pos()
        self.update()

    def mouseReleaseEvent(self, e):
        self.last_pos = None

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_R:
            self.rotation_x = self.rotation_y = 0.0
            self.x_pan = self.y_pan = 0.0
            self.zoom = -5.0
            self.update()
        elif e.key() == Qt.Key_L:
            # Toggle light position
            if self.light_pos.x() > 0:
                self.light_pos = QVector3D(-5.0, 5.0, 5.0)
            else:
                self.light_pos = QVector3D(5.0, 5.0, 5.0)
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
        info_layout.addWidget(QLabel("• L: Toggle Light"))
        control_layout.addWidget(info_frame)

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

from PyQt5.QtGui import QVector3D, QMatrix4x4

class Mesh:
    def __init__(self, vertices, indices):
        self.vertices = vertices
        self.indices = indices
        self.vao = None
        self.vbo = None
        self.ebo = None

class GameObject:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

        self.position = QVector3D(0, 0, 0)
        self.rotation = QVector3D(0, 0, 0) # Euler angles
        self.scale = QVector3D(1, 1, 1)

        # Components
        self.mesh = None
        self.camera = None
        self.script = None

    def get_model_matrix(self):
        model_matrix = QMatrix4x4()
        model_matrix.translate(self.position)
        # Note: Proper rotation would use Quaternions, but for simplicity we use Euler angles
        model_matrix.rotate(self.rotation.x(), 1, 0, 0)
        model_matrix.rotate(self.rotation.y(), 0, 1, 0)
        model_matrix.rotate(self.rotation.z(), 0, 0, 1)
        model_matrix.scale(self.scale)
        return model_matrix

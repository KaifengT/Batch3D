
in vec4 a_Position;

uniform mat4 u_mvpMatrix;

void main() {
    gl_Position = u_mvpMatrix * a_Position;
}
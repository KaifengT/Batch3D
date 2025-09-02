#version 460 core

layout (location=0) in vec4 a_Position;
layout (location=2) in vec3 a_Normal;

uniform float u_pointSize;

uniform mat4 u_mvpMatrix;
uniform mat4 u_mvMatrix;
uniform mat3 u_normalMatrix;

layout (location = 0) out vec4 ViewPos;
layout (location = 1) out vec4 ViewNormal;


void main() {

    gl_PointSize = u_pointSize;
    gl_Position = u_mvpMatrix * a_Position;
    ViewPos = u_mvMatrix * a_Position;
    ViewNormal = vec4(u_normalMatrix * a_Normal, 1.0);

}
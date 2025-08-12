#version 120

attribute vec4 a_Position;
attribute vec3 a_Normal;

uniform mat4 u_ProjMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ModelMatrix;

varying vec4 ViewPos;
varying vec4 ViewNormal;


void main() {
    mat4 MV = u_ViewMatrix * u_ModelMatrix;
    gl_Position = u_ProjMatrix * MV * a_Position;
    ViewPos = MV * a_Position;

    ViewNormal = vec4(normalize(mat3(MV) * a_Normal), 1.0);
}

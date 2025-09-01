#version 460 core

layout (location=0) in vec4 a_Position;
layout (location=2) in vec3 a_Normal;

uniform mat4 u_ProjMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ModelMatrix;
uniform float u_pointSize;

out vec4 ViewPos;
out vec4 ViewNormal;


void main() {

    mat4 MV = u_ViewMatrix * u_ModelMatrix;
    gl_PointSize = u_pointSize;
    gl_Position = u_ProjMatrix * MV * a_Position;
    ViewPos = MV * a_Position;
    ViewNormal = vec4(mat3(transpose(inverse(MV))) * a_Normal, 1.0);


}
#version 330 core



in vec4 ViewPos;
in vec4 ViewNormal;


layout (location = 0) out vec4 PosOut;
layout (location = 1) out vec4 NormalOut;
// out vec4 PosOut;

void main() {
    PosOut = ViewPos;
    NormalOut = ViewNormal;
}
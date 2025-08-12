#version 120
#extension GL_EXT_draw_buffers : require

varying vec4 ViewPos;
varying vec4 ViewNormal;

void main() {
    gl_FragData[0] = ViewPos;
    gl_FragData[1] = ViewNormal;
}
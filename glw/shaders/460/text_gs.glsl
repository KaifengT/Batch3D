
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

out vec2 v_Texcoord;

uniform vec2 u_screenSize;
uniform vec4 u_bearingAndSize = vec4(2.0, 6.0, 5.0, 8.0); // x: bearingX, y: bearingY, z: width, w: height(rows)
uniform float u_advance = 0.0;
uniform float u_fontSize = 10.0;

void main() {
    vec4 pos_clip = gl_in[0].gl_Position;


    vec2 offset[4];
    offset[0] = vec2(u_bearingAndSize[0] + u_advance, u_bearingAndSize[1]);
    offset[1] = vec2(u_bearingAndSize[0] + u_advance, -(u_bearingAndSize[3] - u_bearingAndSize[1]));
    offset[2] = vec2(u_bearingAndSize[0] + u_bearingAndSize[2] + u_advance, u_bearingAndSize[1]);
    offset[3] = vec2(u_bearingAndSize[0] + u_bearingAndSize[2] + u_advance, -(u_bearingAndSize[3] - u_bearingAndSize[1]));

    vec2 texcoord[4];
    texcoord[0] = vec2(0.0, 0.0);
    texcoord[1] = vec2(0.0, 1.0);
    texcoord[2] = vec2(1.0, 0.0);
    texcoord[3] = vec2(1.0, 1.0);

    for (int i = 0; i < 4; i++) {
        gl_Position = pos_clip + vec4((offset[i] / u_screenSize) * 2.0 * pos_clip.w * u_fontSize, 0, 0);

        v_Texcoord = texcoord[i];
        EmitVertex();
    }
    EndPrimitive();
}
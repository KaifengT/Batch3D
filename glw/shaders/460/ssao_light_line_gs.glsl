
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

in vec3 _v_Position[];
in vec3 _v_Normal[];
in vec4 _v_Color[];
in vec2 _v_Texcoord[];
in vec3 _v_WorldSpaceCamPos[];
flat in int _v_simpleRender[];

out vec3 v_Position;
out vec3 v_Normal;
out vec4 v_Color;
out vec2 v_Texcoord;
out vec3 v_WorldSpaceCamPos;
flat out int v_simpleRender;

uniform float u_lineWidth = 2.0;
uniform vec2 u_screenSize;

void main() {
    vec4 pos0_clip = gl_in[0].gl_Position;
    vec4 pos1_clip = gl_in[1].gl_Position;

    vec2 ndc0 = pos0_clip.xy / pos0_clip.w;
    vec2 ndc1 = pos1_clip.xy / pos1_clip.w;

    vec2 screen0 = (ndc0 + 1.0) * 0.5 * u_screenSize;
    vec2 screen1 = (ndc1 + 1.0) * 0.5 * u_screenSize;

    vec2 direction = normalize(screen1 - screen0);
    vec2 offset = vec2(-direction.y, direction.x) * (u_lineWidth * 0.5);

    vec2 ndcOffset = (offset / u_screenSize) * 2.0;

    vec4 corner[4];
    corner[0] = pos0_clip + vec4(ndcOffset * pos0_clip.w, 0, 0); // 左下
    corner[1] = pos0_clip - vec4(ndcOffset * pos0_clip.w, 0, 0); // 右下
    corner[2] = pos1_clip + vec4(ndcOffset * pos1_clip.w, 0, 0); // 左上
    corner[3] = pos1_clip - vec4(ndcOffset * pos1_clip.w, 0, 0); // 右上

    for (int i = 0; i < 4; i++) {
        gl_Position = corner[i];

        v_Position = _v_Position[i % 2];
        v_Normal = _v_Normal[i % 2];
        v_Color = _v_Color[i % 2];
        v_Texcoord = _v_Texcoord[i % 2];
        v_WorldSpaceCamPos = _v_WorldSpaceCamPos[i % 2];
        v_simpleRender = _v_simpleRender[i % 2];

        EmitVertex();
    }
    EndPrimitive();
}
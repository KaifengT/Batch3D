
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

// 接收顶点着色器的输出
in vec3 _v_Position[];
in vec3 _v_Normal[];
in vec4 _v_Color[];
in vec2 _v_Texcoord[];
in vec3 _v_WorldSpaceCamPos[];
flat in int _v_simpleRender[];

// 传递到片元着色器
out vec3 v_Position;
out vec3 v_Normal;
out vec4 v_Color;
out vec2 v_Texcoord;
out vec3 v_WorldSpaceCamPos;
flat out int v_simpleRender;

// 可调线宽（以屏幕像素为单位）
uniform float u_lineWidth = 2.0; // 像素宽度
uniform vec2 u_screenSize;       // 屏幕分辨率，例如 (1920, 1080)

void main() {
    // 获取两个端点在裁剪空间的位置（来自顶点着色器的 gl_Position）
    vec4 pos0_clip = gl_in[0].gl_Position;
    vec4 pos1_clip = gl_in[1].gl_Position;

    // 转换为 NDC（归一化设备坐标）: [-1, 1]
    vec2 ndc0 = pos0_clip.xy / pos0_clip.w;
    vec2 ndc1 = pos1_clip.xy / pos1_clip.w;

    // 转换为屏幕空间（像素坐标）
    vec2 screen0 = (ndc0 + 1.0) * 0.5 * u_screenSize;
    vec2 screen1 = (ndc1 + 1.0) * 0.5 * u_screenSize;

    // 计算线段方向（在屏幕空间）
    vec2 direction = normalize(screen1 - screen0);
    // 垂直方向（用于加宽）
    vec2 offset = vec2(-direction.y, direction.x) * (u_lineWidth * 0.5);

    // 将偏移量从屏幕空间转回 NDC 空间
    vec2 ndcOffset = (offset / u_screenSize) * 2.0; // 因为 NDC 是 [-1,1]，跨度为2

    // 构建四个顶点（加宽线段的四边形）
    vec4 corner[4];
    corner[0] = pos0_clip + vec4(ndcOffset * pos0_clip.w, 0, 0); // 左下
    corner[1] = pos0_clip - vec4(ndcOffset * pos0_clip.w, 0, 0); // 右下
    corner[2] = pos1_clip + vec4(ndcOffset * pos1_clip.w, 0, 0); // 左上
    corner[3] = pos1_clip - vec4(ndcOffset * pos1_clip.w, 0, 0); // 右上

    // 输出三角形带（0,1,2,3）形成两个三角形：(0,1,2) 和 (1,2,3)
    for (int i = 0; i < 4; i++) {
        gl_Position = corner[i];

        // 传递其他属性（插值用）
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
out vec4 FragColor;

uniform vec4 u_contentPixelRect;
uniform float u_maskAlpha;
uniform vec4 u_lineColor;

void main() {
    vec2 p = gl_FragCoord.xy;

    bool inside =
        (p.x >= u_contentPixelRect.x) &&
        (p.x < u_contentPixelRect.z) &&
        (p.y >= u_contentPixelRect.y) &&
        (p.y < u_contentPixelRect.w);

    vec2 line_width = vec2(4.0);

    bool inside_line =
        (p.x >= u_contentPixelRect.x - line_width.x) &&
        (p.x < u_contentPixelRect.z + line_width.x) &&
        (p.y >= u_contentPixelRect.y - line_width.y) &&
        (p.y < u_contentPixelRect.w + line_width.y);

    if (inside) {
        discard;
    }
    else if (inside_line && !inside) {
        FragColor = u_lineColor;
    }
    else {

        FragColor = vec4(0.0, 0.0, 0.0, u_maskAlpha);
    }
}

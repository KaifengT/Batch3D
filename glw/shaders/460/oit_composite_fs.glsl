
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D u_AccumTexture;
uniform sampler2D u_RevealTexture;

void main()
{
    // ivec2 coords = ivec2(gl_FragCoord.xy);
    // vec4 accum = texelFetch(u_AccumTexture, coords, 0);
    // float reveal = texelFetch(u_RevealTexture, coords, 0).r;
    
    vec4 accum = texture(u_AccumTexture, TexCoord);
    float reveal = texture(u_RevealTexture, TexCoord).r;

    if (reveal >= 1.0)
        discard;

    vec3 average_color = accum.rgb / max(accum.a, 0.00001);
    
    // Gamma correction
    // average_color = pow(average_color, vec3(1.0/2.2));

    FragColor = vec4(average_color, 1.0 - reveal);

}

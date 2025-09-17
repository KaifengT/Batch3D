

in vec2 v_Texcoord;

out vec4 FragColor;

uniform sampler2D u_AlbedoTexture;
uniform vec3 u_textColor = vec3(0.8, 0.8, 0.8);

void main() {
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(u_AlbedoTexture, v_Texcoord).r);
    FragColor = sampled * vec4(u_textColor, 1.0);
}
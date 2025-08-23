#version 330 core                                                              

in vec3 a_Position; 

out vec2 TexCoord;

void main()
{          
    gl_Position = vec4(a_Position, 1.0);
    TexCoord = (a_Position.xy + vec2(1.0)) / 2.0;
}

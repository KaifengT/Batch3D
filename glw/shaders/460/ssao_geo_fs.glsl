
in vec4 ViewPos;
in vec4 ViewNormal;

uniform int u_FlatShading;

layout (location = 0) out vec4 PosOut;
layout (location = 1) out vec4 NormalOut;
// out vec4 PosOut;

void main() {
    vec3 normal = normalize(ViewNormal.xyz);
    if (u_FlatShading == 1) {
        vec3 flatNormal = cross(dFdx(ViewPos.xyz), dFdy(ViewPos.xyz));
        if (dot(flatNormal, flatNormal) > 1e-12) {
            normal = normalize(flatNormal);
        }
    }

    PosOut = ViewPos;
    NormalOut = vec4(normal, ViewNormal.w);
}

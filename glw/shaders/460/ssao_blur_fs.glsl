
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D u_AOMap;
uniform sampler2D u_PositionMap;
uniform sampler2D u_NormalMap;
uniform vec2 u_TexelSize;

uniform float u_SpatialSigma;
uniform float u_DepthSigma;
uniform float u_NormalSigma;
uniform int   u_Radius;

float spatialWeight(int dx, int dy) {
    float r2 = float(dx*dx + dy*dy);
    float denom = 2.0 * u_SpatialSigma * u_SpatialSigma;
    return exp(-r2 / denom);
}

bool validGeometry(vec3 pos, vec3 normal) {
    return dot(pos, pos) > 1e-12 && dot(normal, normal) > 1e-12;
}

void main() {
    vec3 centerPos = texture(u_PositionMap, TexCoord).xyz;
    vec3 centerNormal = texture(u_NormalMap, TexCoord).xyz;
    float centerAO = texture(u_AOMap, TexCoord).r;

    if (!validGeometry(centerPos, centerNormal)) {
        FragColor = vec4(1.0);
        return;
    }

    vec3 centerN = normalize(centerNormal);
    float sumW = 0.0;
    float aoAccum = 0.0;

    for (int y = -u_Radius; y <= u_Radius; ++y) {
        for (int x = -u_Radius; x <= u_Radius; ++x) {

            vec2 offset = vec2(x, y) * u_TexelSize;
            vec2 tc = TexCoord + offset;

            vec3 samplePos = texture(u_PositionMap, tc).xyz;
            vec3 sampleNormal = texture(u_NormalMap, tc).xyz;
            float sampleAO = texture(u_AOMap, tc).r;


            if (!validGeometry(samplePos, sampleNormal)) {
                continue;
            }

            vec3 sampleN = normalize(sampleNormal);
            float wSpatial = spatialWeight(x, y);

            float dz = samplePos.z - centerPos.z;
            float wDepth = exp(-(dz*dz) / (2.0 * u_DepthSigma * u_DepthSigma));

            float ndot = max(dot(centerN, sampleN), 0.0);
            float normalDiff = 1.0 - ndot;
            float wNormal = exp(-(normalDiff * normalDiff) * u_NormalSigma);

            float w = wSpatial * wDepth * wNormal;

            aoAccum += sampleAO * w;
            sumW += w;
        }
    }

    float ao = (sumW > 0.0) ? (aoAccum / sumW) : centerAO;
    ao = clamp(ao, 0.0, 1.0);

    FragColor = vec4(ao);
}


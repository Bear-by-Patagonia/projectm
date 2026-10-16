precision mediump float;

in vec4 frag_COLOR;
in vec4 frag_TEXCOORD0;
in vec2 frag_TEXCOORD1;

uniform sampler2D texture_sampler;
uniform int u_catmullRom;

layout(location = 0) out vec4 color;
layout(location = 1) out vec2 texCoords;

// Fast 4-Tap Catmull-Rom Bicubic filtering exploiting GPU bilinear interpolator
vec4 sampleCatmullRom4Tap(sampler2D tex, vec2 uv) {
    vec2 texSize = vec2(textureSize(tex, 0));
    vec2 invTexSize = 1.0 / texSize;
    vec2 samplePos = uv * texSize;
    vec2 tc = floor(samplePos - 0.5) + 0.5;
    vec2 f = samplePos - tc;
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;

    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w3 = 0.5 * (f3 - f2);
    vec2 w2 = 1.0 - w0 - w1 - w3;

    vec2 s0 = w0 + w1;
    vec2 s1 = w2 + w3;

    vec2 f0 = w1 / s0;
    vec2 f1 = w3 / s1;

    vec2 t0 = (tc - 1.0 + f0) * invTexSize;
    vec2 t1 = (tc + 1.0 + f1) * invTexSize;

    return (texture(tex, vec2(t0.x, t0.y)) * (s0.x * s0.y) +
            texture(tex, vec2(t1.x, t0.y)) * (s1.x * s0.y) +
            texture(tex, vec2(t0.x, t1.y)) * (s0.x * s1.y) +
            texture(tex, vec2(t1.x, t1.y)) * (s1.x * s1.y));
}

void main() {
    vec4 texColor;
    if (u_catmullRom != 0) {
        texColor = sampleCatmullRom4Tap(texture_sampler, frag_TEXCOORD0.xy);
    } else {
        texColor = texture(texture_sampler, frag_TEXCOORD0.xy);
    }
    color = frag_COLOR * texColor;
    texCoords = frag_TEXCOORD0.xy;
}

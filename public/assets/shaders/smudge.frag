#include <common>

uniform sampler2D tDiffuse;
uniform sampler2D dmap;
uniform vec2 resolution;
uniform vec2 uDir;
uniform float amp;
uniform float seed;

varying vec2 vUv;

//uniform float sigma;     // The sigma value for the gaussian function: higher value means more blur
                        // A good value for 9x9 is around 3 to 5
                        // A good value for 7x7 is around 2.5 to 4
                        // A good value for 5x5 is around 2 to 3.5
                        // ... play around with this based on what you need :)

//uniform float blurSize;  // This should usually be equal to
                        // 1.0f / texture_pixel_width for a horizontal blur, and
                        // 1.0f / texture_pixel_height for a vertical blur.

const float pi = 3.14159265f;

const float numBlurPixelsPerSide = 4.0f;



float randomNoise(vec2 p) {
return fract(16791.414*sin(7.*p.x+p.y*73.41));
}

float random (in vec2 _st) {
    return fract(sin(dot(_st.xy,
                        vec2(12.9898,78.233)))*
        43758.5453123);
}

float mmap(float v, float v1, float v2, float v3, float v4){
    return (v-v1)/(v2-v1)*(v4-v3)+v3;
}

float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

float noise3 (in vec2 _st, in float t) {
    vec2 i = floor(_st+t);
    vec2 f = fract(_st+t);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

#define NUM_OCTAVES 8

float fbm ( in vec2 _st) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    // Rotate to reduce axial bias
    mat2 rot = mat2(cos(0.5), sin(0.5),
                    -sin(0.5), cos(0.50));
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise(_st);
        _st = rot * _st * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

float fbm3 ( in vec2 _st, in float t) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    // Rotate to reduce axial bias
    mat2 rot = mat2(cos(0.5), sin(0.5),
                    -sin(0.5), cos(0.50));
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise3(_st, t);
        _st = rot * _st * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}


// gaussian blur filter modified from Filip S. at intel 
// https://software.intel.com/en-us/blogs/2014/07/15/an-investigation-of-fast-real-time-gpu-based-image-blur-algorithms
// this function takes three parameters, the texture we want to blur, the uvs, and the texelSize
vec3 gaussianBlur( sampler2D t, vec2 texUV, vec2 stepSize ){   
    // a variable for our output                                                                                                                                                                 
    vec3 colOut = vec3( 0.0 );                                                                                                                                   

    // stepCount is 9 because we have 9 items in our array , const means that 9 will never change and is required loops in glsl                                                                                                                                     
    const int stepCount = 9;

    // these weights were pulled from the link above
    float gWeights[stepCount];
        gWeights[0] = 0.10855;
        gWeights[1] = 0.13135;
        gWeights[2] = 0.10406;
        gWeights[3] = 0.07216;
        gWeights[4] = 0.04380;
        gWeights[5] = 0.02328;
        gWeights[6] = 0.01083;
        gWeights[7] = 0.00441;
        gWeights[8] = 0.00157;

    // these offsets were also pulled from the link above
    float gOffsets[stepCount];
        gOffsets[0] = 0.66293;
        gOffsets[1] = 2.47904;
        gOffsets[2] = 4.46232;
        gOffsets[3] = 6.44568;
        gOffsets[4] = 8.42917;
        gOffsets[5] = 10.41281;
        gOffsets[6] = 12.39664;
        gOffsets[7] = 14.38070;
        gOffsets[8] = 16.36501;
    
    // lets loop nine times
    for( int i = 0; i < stepCount; i++ ){  

        // multiply the texel size by the by the offset value                                                                                                                                                               
        vec2 texCoordOffset = gOffsets[i] * stepSize;

        // sample to the left and to the right of the texture and add them together                                                                                                           
        vec3 col = texture2D( t, texUV + texCoordOffset ).xyz + texture2D( t, texUV - texCoordOffset ).xyz; 

        // multiply col by the gaussian weight value from the array
        col *= gWeights[i];

        // add it all up
        colOut +=  col;                                                                                                                               
    }

    // our final value is returned as col out
    return colOut;                                                                                                                                                   
} 

float fff(vec2 st, float seed){

    vec2 q = vec2(0.);
    q.x = fbm3( st + 0.1, seed*.11);
    q.y = fbm3( st + vec2(1.0), seed*.11);
    vec2 r = vec2(0.);
    r.x = fbm3( st + 1.0*q + vec2(1.7,9.2)+ 0.15*seed*0.11, seed*.11);
    r.y = fbm3( st + 1.0*q + vec2(8.3,2.8)+ 0.126*seed*0.11, seed*.11);
    float f = fbm3(st+r, seed*.11);
    float ff = (f*f*f+0.120*f*f+.5*f);

    return ff;
}

vec4 blur(sampler2D t, vec2 coor, float blurSize, vec2 direction){
    float sigma = 3.0;
    // Incremental Gaussian Coefficent Calculation (See GPU Gems 3 pp. 877 - 889)
    vec3 incrementalGaussian;
    incrementalGaussian.x = 1.0f / (sqrt(2.0f * pi) * sigma);
    incrementalGaussian.y = exp(-0.5f / (sigma * sigma));
    incrementalGaussian.z = incrementalGaussian.y * incrementalGaussian.y;
    
    vec4 avgValue = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    float coefficientSum = 0.0f;
    
    // Take the central sample first...
    avgValue += texture2D(t, coor.xy) * incrementalGaussian.x;
    coefficientSum += incrementalGaussian.x;
    incrementalGaussian.xy *= incrementalGaussian.yz;
    
    // Go through the remaining 8 vertical samples (4 on each side of the center)
    for (float i = 1.0f; i <= numBlurPixelsPerSide; i++) { 
        avgValue += texture2D(t, coor.xy - i * blurSize * 
                            direction) * incrementalGaussian.x;         
        avgValue += texture2D(t, coor.xy + i * blurSize * 
                            direction) * incrementalGaussian.x;         
        coefficientSum += 2. * incrementalGaussian.x;
        incrementalGaussian.xy *= incrementalGaussian.yz;
    }
    
    return avgValue / coefficientSum;
}

float power(float p, float g) {
    if (p < 0.5)
        return 0.5 * pow(2.*p, g);
    else
        return 1. - 0.5 * pow(2.*(1. - p), g);
}

float qnt(float v, float q){
    return floor(v*q)/q;
}

void main_old() {

    vec2 xy = gl_FragCoord.xy;
    vec2 uv = xy / resolution;

    uv = vUv;
    
    vec4 texel = texture2D( tDiffuse, uv);
    vec4 depth = texture2D( dmap, uv);

    float qua = floor(23. + 39.*smoothstep(.4, .75, fff(uv*vec2(0., 1.)*114.4+seed*14., seed)));
    float qua2 = floor(2. + 5.*smoothstep(.4, .75, fff(uv*vec2(0., 1.)*1.4+seed*14., seed)));
    qua = qua2;
    float ff = fff(uv*vec2(1., 1.)*3.+seed*14., seed);
    float ffx = -1. + 2.*smoothstep(0., 1., fff(vec2(qnt(uv.x, qua), uv.y)*vec2(1., 1.)*11.3+seed*3., seed+3.413));
    float ffy = -1. + 2.*smoothstep(0., 1., fff(vec2(qnt(uv.x, qua), uv.y)*vec2(1., 1.)*11.3+seed*2.+31.31, seed+1.413));
    float ffa1 = -1. + 2.*smoothstep(0., 1., fff(vec2(qnt(uv.x, qua), uv.y)*vec2(1., 1.)*11.3+seed*14.+vec2(ffx, ffy), seed+123.41));
    float ffa2 = -1. + 2.*smoothstep(0., 1., fff(vec2(qnt(uv.x, qua), uv.y)*vec2(1., 1.)*11.3+seed*24.+vec2(ffx, ffy), seed+44.13));
    float ffb1 = -1. + 2.*pow(smoothstep(0., 1., fff(vec2(qnt(uv.x, qua), uv.y)*vec2(1., 14.)*1.14+seed*14.+.6*vec2(ffa1, ffa2), seed+11.13)), 1.);
    float ffb2 = -1. + 2.*pow(smoothstep(0., 1., fff(vec2(qnt(uv.x, qua), uv.y)*vec2(1., 14.)*1.14+seed*1.42+.6*vec2(ffa1, ffa2), seed+33.43)), 1.);
    float ffc  = pow(smoothstep(0.3, 0.6, fff(uv*vec2(1., 14.)*1.77+seed*2.67+13.*vec2(ffa1, ffa2), seed+73.77)), 3.);
    float ffc1  = -1. + 2.*pow(smoothstep(0., 1., fff(vec2(qnt(uv.x, qua), uv.y)*vec2(1., 11.)*3.2+seed*3.6877+1.1*vec2(ffb1, ffb2), seed+2.53)), 3.);
    float ffc2  = -1. + 2.*pow(smoothstep(0., 1., fff(vec2(qnt(uv.x, qua), uv.y)*vec2(1., 11.)*3.2+seed*2.67+1.1*vec2(ffb1, ffb2)+38., seed*11.23+7.56)), 3.);

    ffc = ffc*2.-1.;

    float ang = ff*1.;
    float ux = smoothstep(0., 1., ffx);
    float uy = smoothstep(0., 1., ffy);
    vec2 uu = vec2(2.*ux-1., 2.*uy-1.);
    uu /= length(uu);
    uu *= ff;

    if(uDir.x > .5){
        uu = vec2(-uu.y, uu.x);
    }
    ffc = 1.;
    vec2 uvm = uv + ff*.0031*vec2(ffb1, ffb2*.2);

    float ooi = .08 + .92*pow(1.-abs(uv.y-.5)/.5, 4.);

    ooi *= .1;

    vec2 udi = vec2(ffc1*.04, ffc2);

    //ffx = floor(ffx*3.)/3.;
    float ooax = cos(3.14/2. + .12*ffb1*3.14);
    float ooay = sin(3.14/2. + .12*ffb1*3.14);
    udi = vec2(ooax, ooay) * (.3 + .7*smoothstep(.0, .9, ffa1*.5+.5));
    //udi = uDir;

    if(uDir.x < .5){
        udi = vec2(-udi.y, udi.x);
    }
    vec3 colorr, colorg, colorb;

    float posmx = -1.+2.*smoothstep(.0, .9, .5+.5*ffb1);
    float posmy = .1*ffc2;
    vec2 posm = vec2(posmx, posmy);

    //colorr = gaussianBlur(tDiffuse, uv + ooi*ffc*.25*udi, 11.*ooi*ffc*4.*1.40*amp/resolution.x*udi ); //*uu*randomNoise(uv)
    //colorg = gaussianBlur(tDiffuse, uv - 0.*ooi*ffc*.2*udi, 11.*ooi*ffc*4.*1.40*amp/resolution.x*udi ); //*uu*randomNoise(uv)
    //colorb = gaussianBlur(tDiffuse, uv - ooi*ffc*.25*udi, 11.*ooi*ffc*4.*1.40*amp/resolution.x*udi ); //*uu*randomNoise(uv)
    colorr = gaussianBlur(tDiffuse, uv + 33.00*udi*amp/1000. - 0.*ooi*ffc*.2*udi, 11.1*udi*amp/1000. ); //*uu*randomNoise(uv)
    colorg = gaussianBlur(tDiffuse, uv + 33.00*udi*amp/1000. - 0.*ooi*ffc*.2*udi, 11.1*udi*amp/1000. ); //*uu*randomNoise(uv)
    colorb = gaussianBlur(tDiffuse, uv + 33.00*udi*amp/1000. - 0.*ooi*ffc*.2*udi, 11.1*udi*amp/1000. ); //*uu*randomNoise(uv)
    vec3 color = vec3(colorr.r, colorg.g, colorb.b);
    gl_FragColor = vec4(vec3(uu.xy, 0.), 1.);
    gl_FragColor = vec4(vec3(uu*randomNoise(uv), 0.), 1.);
    gl_FragColor = vec4(vec3(.5*ffc+.5), 1.);
    gl_FragColor = vec4(vec3(.5*ffx+.5), 1.);
    gl_FragColor = vec4(vec3(udi.xy*.5+.5, 0.), 1.);
    gl_FragColor = vec4(vec3(.5*ffc2+.5), 1.);
    gl_FragColor = vec4(color.rgb, 1.);

}

void main() {

    vec2 xy = gl_FragCoord.xy;
    vec2 uv = xy / resolution;

    uv = vUv;
    
    vec4 depth = texture2D( dmap, uv);

    vec2 uvq = uv + 20.*1./1000.*(depth.gb-.5);
    //uv = uv + 20.*1./1000.*(depth.gb-.5);

    vec4 texel = texture2D( tDiffuse, uv);

    float ff = fff(uv*vec2(1., 1.)*3.+seed*14., seed);
    float ffx = -1. + 2.*smoothstep(0., 1., fff(uv*vec2(1., 1.)*.3+seed*3., seed+3.413));
    float ffy = -1. + 2.*smoothstep(0., 1., fff(uv*vec2(1., 1.)*.3+seed*2.+31.31, seed+1.413));
    float ffa1 = -1. + 2.*smoothstep(0., 1., fff(uv*vec2(1., 1.)*.3+seed*14.+vec2(ffx, ffy), seed+123.41));
    float ffa2 = -1. + 2.*smoothstep(0., 1., fff(uv*vec2(1., 1.)*.3+seed*24.+vec2(ffx, ffy), seed+44.13));
    float ffb1 = -1. + 2.*pow(smoothstep(0., 1., fff(uv*vec2(1., 6.)*.14+seed*14.+vec2(ffa1, ffa2), seed+11.13)), 1.);
    float ffb2 = -1. + 2.*pow(smoothstep(0., 1., fff(uv*vec2(1., 6.)*.14+seed*1.42+vec2(ffa1, ffa2), seed+33.43)), 1.);
    float ffc  = -1. + 2.*pow(smoothstep(0., 1., fff(uv*vec2(1., 6.)*1.77+seed*2.67+vec2(ffa1, ffa2), seed+73.77)), 3.);
    float ffc1  = -1. + 2.*pow(smoothstep(0., 1., fff(uv*vec2(1., 6.)*7.43+seed*3.6877+vec2(ffb1, ffb2), seed+2.53)), 3.);
    float ffc2  = -1. + 2.*pow(smoothstep(0., 1., fff(uv*vec2(1., 6.)*1.21+seed*2.67+vec2(ffb1, ffb2), seed*1.23+7.56)), 3.);


    float ang = ff*1.;
    float ux = smoothstep(0., 1., ffx);
    float uy = smoothstep(0., 1., ffy);
    vec2 uu = vec2(2.*ux-1., 2.*uy-1.);
    uu /= length(uu);
    uu *= ff;

    if(uDir.x > .5){
        uu = vec2(-uu.y, uu.x);
    }

    vec2 uvm = uv + ff*.0031*vec2(ffb1, ffb2*.2);

    float ooi;
    ooi = .55 + .45*sin(uv.y*2.*3.14*(.3 + 1.7*seed) + random(vec2(seed, seed)));
    ooi = .1 + .9*pow(1.-abs(uv.y-.5)/.5, 2.);

    vec2 udi = vec2(ffc1*(-1. + 2.*power(seed, .4)), ffc2);
    //udi = vec2(ffc1*mmap(smoothstep(.3, .7, fff(uv*13.2+seed*12.34, 3.*seed+34.31)), 0., 1., 0.01, 0.22), ffc2);
    udi = vec2(ffc1*.1, ffc2);
    //udi = uDir;

    ffc *= .2 + .5*smoothstep(.45, .55, seed);

    ffc *= .3 + 1.7*power(1.-abs(uv.y-.5)*2., 3.);

    float ui = 1.;
    if(seed > .5)
        ui = -ui;
    //ffc *= uv.x;
    //ffc *= pow(depth.r, .15);
    vec3 colorr = gaussianBlur(tDiffuse, uv + ui*ooi*ffc*.08*udi, 18.5*ooi*ffc*4.*1.40*amp/1000.*udi ); //*uu*randomNoise(uv)
    vec3 colorg = gaussianBlur(tDiffuse, uv + ooi*0.*ffc*.08*udi, 18.5*ooi*ffc*4.*1.40*amp/1000.*udi ); //*uu*randomNoise(uv)
    vec3 colorb = gaussianBlur(tDiffuse, uv - ui*ooi*ffc*.08*udi, 18.5*ooi*ffc*4.*1.40*amp/1000.*udi ); //*uu*randomNoise(uv)
    vec3 color = vec3(colorr.r, colorg.g, colorb.b);

    gl_FragColor = vec4(vec3(uu.xy, 0.), 1.);
    gl_FragColor = vec4(vec3(uu*randomNoise(uv), 0.), 1.);
    gl_FragColor = vec4(depth.rgb, 1.);
    gl_FragColor = vec4(color.rgb, 1.);

}


void main_good() {

    vec2 xy = gl_FragCoord.xy;
    vec2 uv = xy / resolution;

    uv = vUv;
    
    vec4 texel = texture2D( tDiffuse, uv);
    vec4 texel2 = texture2D( dmap, uv);

    float ff = fff(uv*vec2(1., 1.)*3.+seed*14., seed);
    float ffx = -1. + 2.*smoothstep(0., 1., fff(uv*vec2(1., 1.)*.3+seed*3., seed+3.413));
    float ffy = -1. + 2.*smoothstep(0., 1., fff(uv*vec2(1., 1.)*.3+seed*2.+31.31, seed+1.413));
    float ffa1 = -1. + 2.*smoothstep(0., 1., fff(uv*vec2(1., 1.)*.3+seed*14.+vec2(ffx, ffy), seed+123.41));
    float ffa2 = -1. + 2.*smoothstep(0., 1., fff(uv*vec2(1., 1.)*.3+seed*24.+vec2(ffx, ffy), seed+44.13));
    float ffb1 = -1. + 2.*pow(smoothstep(0., 1., fff(uv*vec2(1., 18.)*.14+seed*14.+vec2(ffa1, ffa2), seed+11.13)), 1.);
    float ffb2 = -1. + 2.*pow(smoothstep(0., 1., fff(uv*vec2(1., 18.)*.14+seed*1.42+vec2(ffa1, ffa2), seed+33.43)), 1.);
    float ffc  = -1. + 2.*pow(smoothstep(0., 1., fff(uv*vec2(1., 18.)*1.77+seed*2.67+vec2(ffa1, ffa2), seed+73.77)), 3.);
    float ffc1  = -1. + 2.*pow(smoothstep(0., 1., fff(uv*vec2(1., 18.)*7.43+seed*3.6877+vec2(ffb1, ffb2), seed+2.53)), 3.);
    float ffc2  = -1. + 2.*pow(smoothstep(0., 1., fff(uv*vec2(1., 18.)*1.21+seed*2.67+vec2(ffb1, ffb2), seed+7.56)), 3.);


    float ang = ff*1.;
    float ux = smoothstep(0., 1., ffx);
    float uy = smoothstep(0., 1., ffy);
    vec2 uu = vec2(2.*ux-1., 2.*uy-1.);
    uu /= length(uu);
    uu *= ff;

    if(uDir.x > .5){
        uu = vec2(-uu.y, uu.x);
    }

    vec2 uvm = uv + ff*.0031*vec2(ffb1, ffb2*.2);

    float ooi = .05 + .95*pow(1.-abs(uv.y-.5)/.5, 4.);

    vec3 colorr = gaussianBlur(tDiffuse, uv + ooi*ffc*.020*vec2(ffc1*.1, ffc2), ooi*ffc*4.*1.40*amp/resolution.x*vec2(ffc1*.1, ffc2) ); //*uu*randomNoise(uv)
    vec3 colorg = gaussianBlur(tDiffuse, uv + ooi*0.*ffc*.020*vec2(ffc1*.1, ffc2), ooi*ffc*4.*1.40*amp/resolution.x*vec2(ffc1*.1, ffc2) ); //*uu*randomNoise(uv)
    vec3 colorb = gaussianBlur(tDiffuse, uv - ooi*ffc*.020*vec2(ffc1*.1, ffc2), ooi*ffc*4.*1.40*amp/resolution.x*vec2(ffc1*.1, ffc2) ); //*uu*randomNoise(uv)
    vec3 color = vec3(colorr.r, colorg.g, colorb.b);
    gl_FragColor = vec4(vec3(uu.xy, 0.), 1.);
    gl_FragColor = vec4(vec3(uu*randomNoise(uv), 0.), 1.);
    gl_FragColor = vec4(vec3(ffb1), 1.);
    gl_FragColor = vec4(texel.rgb, 1.);
    gl_FragColor = vec4(color.rgb, 1.);

}
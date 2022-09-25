import * as THREE from 'three';

import ClipperLib from 'js-clipper';

import {EffectComposer} from 'three/examples/jsm/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/examples/jsm/postprocessing/RenderPass.js';
import {ShaderPass} from 'three/examples/jsm/postprocessing/ShaderPass.js';
import {FilmPass} from 'three/examples/jsm/postprocessing/FilmPass.js';
import {mergeBufferGeometries} from 'three/examples/jsm/utils/BufferGeometryUtils.js';

import {FXAAShader} from 'three/examples/jsm/shaders/FXAAShader.js';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';

let scene;
let camera;
let realCamera;
let renderer;
let camcontrols;
let composer;

let displayTarget;

let display;


const rybCol = require("./artcolors").rybCol;
const noise = require("./noise").noise;

function map(v, v1, v2, v3, v4){
    return (v-v1)/(v2-v1)*(v4-v3)+v3;
}


let baseMaterial = new THREE.ShaderMaterial({
    lights: true,
    side: THREE.DoubleSide,
    uniforms: {
        ...THREE.UniformsLib.lights,
        color: {
            value: new THREE.Color(...rybCol(Math.pow(fxrand(), 1)))
        },
        color2: {
            value: new THREE.Color(...rybCol(Math.pow(fxrand(), 1)))
        },
        resolution: {
            value: [100, 100]
        },
        seed: {
            value: fxrand()
        },
        cameraPos: {
            value: [0,0,0]
        }
    },
    vertexShader: `
        #include <common>
        #include <shadowmap_pars_vertex>

        attribute vec4 color;

        varying vec3 vNormal;
        varying vec3 vViewDir;
        varying vec4 vColor;
        varying vec2 vUV;
        varying vec4 modelPos;
        varying vec3 vWNormal;


        void main() {
            #include <beginnormal_vertex>
            #include <defaultnormal_vertex>

            #include <begin_vertex>

            #include <worldpos_vertex>
            #include <shadowmap_vertex>

            vec4 modelPosition = modelMatrix * vec4(position, 1.0);
            vec4 viewPosition = viewMatrix * modelPosition;
            vec4 clipPosition = projectionMatrix * viewPosition;

            vNormal = normalize(normalMatrix * normal);
            vViewDir = normalize(-viewPosition.xyz);
            vColor = color;
            vUV.xy = uv.xy;
            modelPos = modelPosition;
            vWNormal = normal;

            gl_Position = clipPosition;

        }`
    ,
    fragmentShader: `
        #include <common>
        #include <packing>
        #include <lights_pars_begin>
        #include <shadowmap_pars_fragment>
        #include <shadowmask_pars_fragment>

        uniform vec3 color;
        uniform vec3 color2;
        uniform vec2 resolution;
        uniform float seed;

        varying vec3 vNormal;
        varying vec3 vWNormal;
        varying vec3 vViewDir;
        varying vec4 vColor;
        varying vec2 vUV;
        varying vec4 modelPos;

        uniform vec3 cameraPos;

        const float pi = 3.14159265f;
        const float numBlurPixelsPerSide = 4.0f;

        float power(float p, float g) {
            if (p < 0.5)
                return 0.5 * pow(2.*p, g);
            else
                return 1. - 0.5 * pow(2.*(1. - p), g);
        }
        
        float hash13(vec3 p3)
        {
            p3  = fract(p3 * .1031);
            p3 += dot(p3, p3.zyx + 31.32);
            return fract((p3.x + p3.y) * p3.z);
        }

        void main() {
            // shadow map
            DirectionalLightShadow directionalLight = directionalLightShadows[0];

            float shadow = getShadow(
                directionalShadowMap[0],
                directionalLight.shadowMapSize,
                directionalLight.shadowBias,
                directionalLight.shadowRadius,
                vDirectionalShadowCoord[0]
            );
            shadow = smoothstep(.1, .2, shadow);

            // float NdotL = dot(vNormal, directionalLights[0].direction);
            // float lightIntensity = smoothstep(0.0, 0.01, NdotL * 1.);
            // vec3 light = directionalLights[0].color * lightIntensity;

            // vec3 halfVector = normalize(directionalLights[0].direction + vViewDir);
            // float NdotH = dot(vNormal, halfVector);

            // float specularIntensity = pow(NdotH * lightIntensity, uGlossiness * uGlossiness);
            // float specularIntensitySmooth = smoothstep(0.05, 0.1, specularIntensity);

            // vec3 specular = specularIntensitySmooth * directionalLights[0].color;

            // float rimDot = 1.0 - dot(vViewDir, vNormal);
            // float rimAmount = 0.6;

            // float rimThreshold = 0.2;
            // float rimIntensity = rimDot * pow(NdotL, rimThreshold);
            // rimIntensity = smoothstep(rimAmount - 0.01, rimAmount + 0.01, rimIntensity);

            // vec3 rim = rimIntensity * directionalLights[0].color;
            shadow = .5 + .5*shadow;

            vec3 scolor = color;

            float fac = 1.;
            if(vWNormal.y < .5){
                fac = (.6+.4*vUV.y);
                scolor = mix(mix(color, vec3(1.), .0), color2, vUV.y + .0*hash13(vec3(vUV.xy*1000., 1.)));
            }
            if(modelPos.y < .06){
                scolor = vec3(1.,0.,0.);
            }

            scolor = scolor*shadow + vec3(.7,0.,0.)*(1.-shadow);

            scolor = scolor * fac;
            vec3 outcolor = scolor*1.1;

            float aa = length(cameraPos)*3.;
            float toCam = length(cameraPos - modelPos.xyz)/aa;
            vec3 fog = mix(scolor, vec3(.2, .2, .3), smoothstep(.0, .4, toCam));

            gl_FragColor = vec4(vec3(vWNormal.xyz), 1.0);
            gl_FragColor = vec4(outcolor, 1.0);
            gl_FragColor = vec4(vec3(toCam), 1.0);
            gl_FragColor = vec4(fog, 1.0);

        }`
    ,
    transparent: true,
});

let cx = 0;
let cy = 0;
let cz = 0;

let displayMaterial = new THREE.ShaderMaterial({
    uniforms: {
        'sceneRender': {
            value: null
        },
        color: {
            value: new THREE.Color(...rybCol(Math.pow(fxrand(), 1)))
        },
        resolution: {
            value: [100, 100]
        },
        seed: {
            value: fxrand()
        }
    },
    vertexShader: `
        varying vec3 vUV; 

        void main() {
            vUV = uv.xyx; 
            vUV.xy = vUV.xy;
            vUV = vUV;

            vec4 modelViewPosition = modelViewMatrix * vec4(position, 1.0);
            gl_Position = projectionMatrix * modelViewPosition; 
        }
    `,
    fragmentShader: `
        uniform vec3 colorA; 
        uniform vec3 colorB; 
        varying vec3 vUV;
        uniform sampler2D sceneRender;
        uniform float seed;
        
        
        float hash13(vec3 p3)
        {
            p3  = fract(p3 * .1031);
            p3 += dot(p3, p3.zyx + 31.32);
            return fract((p3.x + p3.y) * p3.z);
        }

        vec3 rgb2hsv(vec3 c)
        {
            vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
            vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
            vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
        
            float d = q.x - min(q.w, q.y);
            float e = 1.0e-10;
            return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
        }
        
        float power(float p, float g) {
            if (p < 0.5)
                return 0.5 * pow(2.*p, g);
            else
                return 1. - 0.5 * pow(2.*(1. - p), g);
        }

        vec3 hsv2rgb(vec3 c)
        {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        
        float aspect = 1.0;
        float distortion = 0.14;
        float radius = 1.0;
        float alpha = 1.0;
        float crop = 1.0;
        vec2 distort(vec2 p)
        {
            p = p - .5;
            float d = length(p);
            float z = sqrt(distortion + d * d * -distortion);
            float r = d*d/atan(d, z) / 3.1415926535 * 6.;
            float phi = atan(p.y, p.x);
            return (vec2(r * cos(phi) * (1.0 / aspect), r * sin(phi))+.5)*.05 + .95*(p+.5);
        }

        void main() {

            vec2 uv = gl_FragCoord.xy;
            float freq = 300.;

            float scanlinesx = .5 + .5*cos(vUV.x*3.14*freq);
            float scanlinesy = .5 + .5*cos(vUV.y*3.14*freq);
            scanlinesx = .95 + (1.-.95)*scanlinesx;
            scanlinesy = .95 + (1.-.95)*scanlinesy;
            float scanlines = scanlinesx * scanlinesy;

            float vignette = pow(length(vUV.xy-.5)/sqrt(.5), 3.);

            vec2 distorted = distort(vUV.xy);

            // vum.x = floor(vUV.x*(freq))/(freq);
            // vum.y = floor(vUV.y*freq)/freq;
            // vec2 vumq = vum;

            // vum = (vUV.xy - vum)*freq;
            // vum.x = power(vum.x*.99+.001, 3.4);
            // vum.y = power(vum.y*.99+.001, 3.4);
            // vum = vum/freq + vumq;
            // float vd = length(vum.xy);
            // float ang = atan(vum.y, vum.x);
            // float sqrt5 = sqrt(.5);
            // vd = pow(vd/sqrt5, 1.33)*sqrt5;
            // vum.x = vd*cos(ang);
            // vum.y = vd*sin(ang);
            // vum = vum + .5;

            vec2 vum = distorted;

            vec3 res0 = texture2D(sceneRender, vum.xy).rgb;

            vec3 resr = texture2D(sceneRender, vum.xy + 1./555.*vec2(+1., 0.) + res0.r*0./555.*hash13(vec3(vUV.xy*200., seed))).rgb;
            vec3 resg = texture2D(sceneRender, vum.xy + 1./555.*vec2( 0., 0.)).rgb;
            vec3 resb = texture2D(sceneRender, vum.xy + 1./555.*vec2(-1., 0.)).rgb;
            vec3 res = vec3(resr.r, resg.g, resb.b);

            res = rgb2hsv(res);
            res.g *= .65;
            res.b = power(res.b, 3.);
            res = hsv2rgb(res);
            res = mix(res, vec3(0.,0.4,1.), .1);
             res = res * scanlines;

            res = res + vignette*.4;

            res += .025*(-.1 + 2.*hash13(vec3(uv, seed)));

            gl_FragColor = vec4(res.rgb, 1.0);
            // gl_FragColor = vec4(vec3(distorted.x, distorted.y, 0.), 1.0);
        }
    `,
});

let bgColor = new THREE.Color(...rybCol(.5+.1*fxrand()));
bgColor.offsetHSL(0,-.0,-.25);
bgColor = new THREE.Color(.05,.05,.05) 
bgColor = new THREE.Color(.2, .2, .3) 

function setup(){
    scene = new THREE.Scene();
    scene.background = bgColor;
    camera = new THREE.PerspectiveCamera( 55, window.innerWidth / window.innerHeight, 0.1, 1000 );
    realCamera = new THREE.PerspectiveCamera( 44, 4/3, 0.01, 1000 );
    
    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize( window.innerWidth, window.innerHeight );
    renderer.setPixelRatio( window.devicePixelRatio );
    document.body.appendChild( renderer.domElement );
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap


    window.addEventListener( 'resize', onWindowResize );
    displayTarget = new THREE.WebGLRenderTarget(1000, 700);


    composer = new EffectComposer( renderer );


    for(let k = 0; k < 7; k++){

        let x = map(fxrand(), 0, 1, -10, 10);
        let z = map(fxrand(), 0, 1, -10, 10);
        let w = map(fxrand(), 0, 1, 1, 4);
        let d = map(fxrand(), 0, 1, 1, 4);
        let h = map(fxrand(), 0, 1, 1, 14);

        let geometry = new THREE.BoxGeometry( w, h, d );

        // var uvAttribute = geometry.attributes.uv;
        // geometry.attributes.uv.name = "uvs";
        // console.log(uvAttribute)
        // for ( var i = 0; i < uvAttribute.count; i ++ ) {
        //     var u = uvAttribute.getX( i );
        //     var v = uvAttribute.getY( i );
        //     uvAttribute.setXY( i, u, v );
        // }

        let material = baseMaterial.clone();
        material.uniforms.color.value = rybCol(Math.pow(fxrand(), 3));
        material.uniforms.color2.value = rybCol(Math.pow(fxrand(), 1));
        let cube = new THREE.Mesh( geometry, material );
        cube.castShadow = true;
        cube.receiveShadow = true;
        cube.position.x = x;
        cube.position.z = z;
        cube.position.y = h/2;
        objects.push(cube);
        scene.add( cube );
    }

    setupFloor();
    
    setupDisplay();

    addLight();

    setupCam();

    let renderPass = new RenderPass( scene, camera );
    composer.addPass( renderPass );

    
    let fxaaPass = new ShaderPass( FXAAShader );
    fxaaPass.material.uniforms[ 'resolution' ].value.x = 1 / ( window.innerWidth * window.devicePixelRatio );
    fxaaPass.material.uniforms[ 'resolution' ].value.y = 1 / ( window.innerHeight * window.devicePixelRatio );
    composer.addPass( fxaaPass );
    
}

function onWindowResize(){
    // window.cancelAnimationFrame(true);
    // setup();
    // animate();
}

function repositionCanvas(canvas) {
    
}

function setupDisplay(){
    
    let displayGeo = new THREE.PlaneGeometry( 4/40.5, 3/40.5 );
    display = new THREE.Mesh( displayGeo, displayMaterial );
    scene.add( display )
}

let objects = [];

function setupFloor(){
    let planeGeometry = new THREE.BoxGeometry( 27, .1, 27 );
    planeGeometry.computeVertexNormals();
    let planeMaterial = baseMaterial.clone();
    planeMaterial.uniforms.color.value = rybCol(Math.pow(fxrand(), 3));
    let plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.receiveShadow = true;
    plane.castShadow = true;
    objects.push(plane);
    scene.add( plane )
}

function setupCam(){
    cx = 25;
    cy = 25;
    cz = 25;
    camera.position.x = cx;
    camera.position.y = cy;
    camera.position.y = cz;
    
    camcontrols = new OrbitControls( camera, renderer.domElement );
    camcontrols.update();
}

function power(p, g) {
    if (p < 0.5) 
        return 0.5 * Math.pow(2 * p, g);
    else 
        return 1 - 0.5 * Math.pow(2 * (1 - p), g);
}


function addLight(){
    let light = new THREE.DirectionalLight( 0xffffff, 1 );
    light.position.set( 17, 17, 17 ); //default; light shining from top
    light.castShadow = true; // default false
    scene.add( light );
    
    //Set up shadow properties for the light
    light.shadow.mapSize.width = 2000; // default
    light.shadow.mapSize.height = 2000; // default
    light.shadow.camera.near = 0.5; // default
    light.shadow.camera.far = 50; // default
    light.shadow.camera.top = 20; // default
    light.shadow.camera.bottom = -20; // default
    light.shadow.camera.right = -20; // default
    light.shadow.camera.left = 20; // default
    //let helper = new THREE.CameraHelper( light.shadow.camera );
    //scene.add( helper );
}

let frameCount = 0;

function animate() {
    requestAnimationFrame( animate );

    for(let obj of objects){
        obj.material.uniforms.cameraPos.value = [camera.position.x, camera.position.y, camera.position.z]
    }


    var vector = new THREE.Vector3( 0, 0, -1 );
    vector.applyQuaternion( camera.quaternion );
    vector.normalize();

    var right = new THREE.Vector3( 0, 0, - 1 );
    right.crossVectors(vector, new THREE.Vector3( 0, 1, 0 ))
    right.normalize();
    right.multiplyScalar(0.12 * window.innerWidth/2000);

    var down = new THREE.Vector3( 0, 0, - 1 );
    down.crossVectors(vector, right)
    down.normalize();
    down.multiplyScalar(0.05 * window.innerWidth/2000);
    //vector.cross(camera.up);

    cx = cx + .1*(camera.position.x - cx);
    cy = cy + .1*(camera.position.y - cy);
    cz = cz + .1*(camera.position.z - cz);

    let amp = Math.sqrt((camera.position.x - cx)**2 + (camera.position.y - cy)**2 + (camera.position.z - cz)**2);

    
    realCamera.position.x = camera.position.x + 4*vector.x + right.x;
    realCamera.position.y = camera.position.y + 4*vector.y + right.y;
    realCamera.position.z = camera.position.z + 4*vector.z + right.z;
    realCamera.rotation.x = camera.rotation.x + (.3 + .7*amp)*.05*(-.5 + power(noise(frameCount*0.03, .213), 3));
    realCamera.rotation.y = camera.rotation.y + (.3 + .7*amp)*.05*(-.5 + power(noise(frameCount*0.03, 12.55), 3));
    realCamera.rotation.z = camera.rotation.z + (.3 + .7*amp)*.05*(-.5 + power(noise(frameCount*0.01, 42.6), 3));

    
    display.position.x = camera.position.x + .2*vector.x + down.x + right.x + right.x*(.3 + .7*amp)*.015*(-.5 + power(noise(frameCount*0.03, 6.65), 3));
    display.position.y = camera.position.y + .2*vector.y + down.y + right.y + right.y*(.3 + .7*amp)*.15*(-.5 + power(noise(frameCount*0.03, 31.56), 3));
    display.position.z = camera.position.z + .2*vector.z + down.z + right.z + right.z*(.3 + .7*amp)*.015*(-.5 + power(noise(frameCount*0.03, 17.25), 3));
    display.rotation.x = realCamera.rotation.x;
    display.rotation.y = realCamera.rotation.y;
    display.rotation.z = realCamera.rotation.z;
    //realCamera.lookAt(scene.position);

    camcontrols.update();
    //renderer.render( scene, camera );

    renderer.setRenderTarget(displayTarget);
    renderer.render( scene, realCamera );
    display.material.uniforms.sceneRender.value = displayTarget.texture;
    display.material.uniforms.seed.value = fxrand();

    renderer.setRenderTarget(null);

    composer.render();
    frameCount++;
};

setup();
animate();
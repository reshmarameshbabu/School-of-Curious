# SURFACE AREA OF SOLIDS USING AR.js AND aframe

## **Concept:**
- AR.js is an efficient Augmented Reality solution on the Web.
- AR.js uses various concepts like a-frame, a-scene, a-entity, a-asset, a-text, a-image etc.

## **Approach:**
- AR.js uses concepts such as marker based AR, you can print your own marker(here we use hiro markers) and detect its position in the camera's video stream and lay your own 3D models on top of it
- It is built on top of Web technologies, right from your own browser, requiring no additional plugins or dependencies.
- A-Frame is an emerging technology from Mozilla, which allows you to create 3D Scenes and Virtual Reality experiences with just a few HTML tags. 
- It’s built on top of WebGL, Three.js and Custom Elements, a part of the emerging HTML Components standard.
- For development, we use a live-server as a simple HTTP server with built-in live reloading 
- WebAR support is added by loading three.ar.js and aframe-ar and adding the ar attribute to the scene. 

## **Aim:**
- On pointing the web camera at the hiro marker after running it on a localhost simple HTTP browser, we obtain the shapes of the object.
- On clicking the next arrow, shape changes from cube to sphere to cylinder. Opacity and position can be changed accordingly.
- On clicking the solid, information about the solid is displayed: type of solid, side shape, number of vertices, faces and edges, and real examples of the solid
- On clicking on volume, the surface area formula of the solid is displayed along with explanation
- All this is integrated into the code by using simple html and AR.js and aframe using entities, objects and images.


## **Code:**
- Create a new html file with the following code
```html
 <!-- include a-frame -->
 <script src="https://aframe.io/releases/0.8.0/aframe.min.js"></script>
 <script src="https://rawgit.com/donmccurdy/aframe-extras/master/dist/aframe-extras.loaders.min.js"></script>
 
 <!-- include ar.js for aframe -->
 <script src='../build/aframe-ar.js'></script>
 <script type="text/javascript">
 // AFRAME.registerComponent('markerhandler', {
 //
 //     init: function() {
 //         const animatedMarker = document.querySelector("#animated-marker");
 //         animatedMarker.addEventListener('click', function(ev){
 //             if (animatedMarker.object3D.visible == true && ev.detail.cursorEl) {
 //                 const entity = document.querySelector('#animated-model');
 //                 entity.setAttribute('src', '#canvas1')
 //             }
 //         });
 // }
 // });
 
 flag=1
 AFRAME.registerComponent('markerhandler', {
 
     init: function() {
         const animatedMarker = document.querySelector("#spark");
         //defining all entities and posters
         const entity1 = document.querySelector('#cube1');
         const entity2 = document.querySelector('#sphere1');
         const entity3 = document.querySelector('#cylinder1');
         const entity4 = document.querySelector('#cube2');
         const entity5 = document.querySelector('#sphere2');
         const entity6 = document.querySelector('#cylinder2');
         const entity7 = document.querySelector('#line1');
         const entity8 = document.querySelector('#line2');
         const entity9 = document.querySelector('#line3');
         const entity10 = document.querySelector('#line4');
         const poster1 = document.querySelector('#cubed');
         const poster2 = document.querySelector('#sphered');
         const poster3 = document.querySelector('#cylindered');
         const poster4 = document.querySelector('#surfacearea');
         const poster5 = document.querySelector('#cubeans');
         const poster6 = document.querySelector('#sphereans');
         const poster7 = document.querySelector('#cylinderans');
         const poster8 = document.querySelector('#a')
         const poster9 = document.querySelector('#r')
         const poster10 = document.querySelector('#h')
         //setting visibility attributes of entities and posters
         //flag counts number of times clicked
         animatedMarker.addEventListener('click', function(ev){
             entity10.setAttribute('visible','false')
             if (animatedMarker.object3D.visible == true && ev.detail.cursorEl) {
                 if(flag==2)
                 {
                   entity2.setAttribute('visible', 'false')
                   entity5.setAttribute('visible','false')
                   entity8.setAttribute('visible','false')
                   entity9.setAttribute('visible','true')
                   entity10.setAttribute('visible','true')
                   entity3.setAttribute('visible', 'true')
                   poster4.setAttribute('visible','true')
                   poster9.setAttribute('visible','true')
                   poster10.setAttribute('visible','true')
                   flag=3
                 }
                 if(flag==1)
                 {
                   entity1.setAttribute('visible', 'false')
                   entity4.setAttribute('visible','false')
                   entity7.setAttribute('visible','false')
                   poster8.setAttribute('visible','false')
                   poster9.setAttribute('visible','true')
                   entity8.setAttribute('visible','true')
                   entity2.setAttribute('visible', 'true')
                   poster4.setAttribute('visible','true')
                   flag=2
                 }
                 if(flag==0)
                 {
                   entity3.setAttribute('visible', 'false')
                   entity6.setAttribute('visible','false')
                   entity9.setAttribute('visible','false')
                   entity10.setAttribute('visible','false')
                   poster9.setAttribute('visible','false')
                   poster10.setAttribute('visible','false')
                   entity1.setAttribute('visible', 'true')
                   entity7.setAttribute('visible','true')
                   poster4.setAttribute('visible','true')
                   poster8.setAttribute('visible','true')
                   flag=1
                 }
                 if(flag==3)
                 {
                   flag=0
                 }
                 poster1.setAttribute('visible','false')
                 poster2.setAttribute('visible','false')
                 poster3.setAttribute('visible','false')
                 poster4.setAttribute('visible','true')
                 poster5.setAttribute('visible','false')
                 poster6.setAttribute('visible','false')
                 poster7.setAttribute('visible','false')
                 console.log(flag)
             }
             });
             //setting visibility attributes of animated markers
             const animatedMarker1 = document.querySelector("#cube1");
             animatedMarker1.addEventListener('click', function(ev1){
                 if (animatedMarker1.object3D.visible == true && ev1.detail.cursorEl) {
                   entity1.setAttribute('visible','false')
                   entity6.setAttribute('visible','false')
                   poster3.setAttribute('visible','false')
                   entity4.setAttribute('visible','true')
                   poster4.setAttribute('visible','true')
                 }
             });
             const animatedMarker2 = document.querySelector("#sphere1");
             animatedMarker2.addEventListener('click', function(ev2){
                 if (animatedMarker2.object3D.visible == true && ev2.detail.cursorEl) {
                   entity2.setAttribute('visible','false')
                   entity4.setAttribute('visible','false')
                   poster1.setAttribute('visible','false')
                   entity5.setAttribute('visible','true')
                 }
             });
             const animatedMarker3 = document.querySelector("#cylinder1");
             animatedMarker3.addEventListener('click', function(ev3){
                 if (animatedMarker3.object3D.visible == true && ev3.detail.cursorEl) {
                   entity3.setAttribute('visible','false')
                   entity5.setAttribute('visible','false')
                   poster2.setAttribute('visible','false')
                   entity6.setAttribute('visible','true')
                 }
             });
             const animatedMarker4 = document.querySelector('#cube2');
             animatedMarker4.addEventListener('click',function(ev4){
                 if(animatedMarker4.object3D.visible == true && ev4.detail.cursorEl){
                    poster3.setAttribute('visible','false')
                    poster1.setAttribute('visible', 'true')
                 }
             });
             const animatedMarker5 = document.querySelector('#sphere2');
             animatedMarker5.addEventListener('click',function(ev5){
                 if(animatedMarker5.object3D.visible == true && ev5.detail.cursorEl){
                    poster1.setAttribute('visible','false')
                    poster2.setAttribute('visible', 'true')
                 }
             });
             const animatedMarker6 = document.querySelector('#cylinder2');
             animatedMarker6.addEventListener('click',function(ev6){
                 if(animatedMarker6.object3D.visible == true && ev6.detail.cursorEl){
                    poster2.setAttribute('visible','false')
                    poster3.setAttribute('visible','true')
                 }
             });
             const animatedMarker7 = document.querySelector('#surfacearea');
             animatedMarker7.addEventListener('click',function(ev7){
                 if(animatedMarker7.object3D.visible == true && ev7.detail.cursorEl){
                     poster4.setAttribute('visible','true')
                     poster5.setAttribute('visible','true')
                 }
             });
            const animatedMarker8 = document.querySelector('#surfacearea');
            animatedMarker8.addEventListener('click',function(ev8){
                if(animatedMarker5.object3D.visible == true && ev8.detail.cursorEl){
                    poster5.setAttribute('visible','false')
                    poster6.setAttribute('visible','true')
                }
            });
            const animatedMarker9 = document.querySelector('#surfacearea');
            animatedMarker9.addEventListener('click',function(ev9){
                if(animatedMarker6.object3D.visible == true && ev9.detail.cursorEl){
                    poster5.setAttribute('visible','false')
                    poster6.setAttribute('visible','false')
                    poster7.setAttribute('visible','true')
                }
            });
 }
 });
 </script>
 <script>ARjs.Context.baseURL = '../../three.js/'</script>
 
 <!-- start the body of your page -->
 
     <!-- Define your 3d scene and enabled ar.js -->
     <a-scene embedded arjs='sourceType: webcam; debugUIEnabled: true;'>
 
         <a-assets>
             <a-asset-item id="animated-asset" src="https://raw.githubusercontent.com/nicolocarpignoli/nicolocarpignoli.github.io/master/ar-playground/models/CesiumMan.gltf"></a-asset-item>
             <img id="canvas" src="../../data/images/spark/next.png">
             <img id="canvas1" src="../../data/images/spark/cube-info.png">
             <img id="canvas2" src="../../data/images/spark/sphere-info.png">
             <img id="canvas3" src="../../data/images/spark/cylinder-info.png">
             <img id="canvas4" src="../../data/images/sa.png">
             <img id="canvas5" src="../../data/images/saans.jpeg">
             <img id="canvas6" src="../../data/images/sphereans.jpeg">
             <img id="canvas7" src="../../data/images/cylinderans.png">
             <img id="canvas8" src="../../data/images/a.png">
             <img id="canvas9" src="../../data/images/r.png">
             <img id="canvas10" src="../../data/images/h.png">
         </a-assets>
 
         <a-marker markerhandler emitevents="true" type="area" cursor="rayOrigin: mouse" preset="hiro" id="animated-marker">
             <!-- <a-entity
                 id="animated-model"
                 gltf-model="#animated-asset"
                 scale="2">
             </a-entity> -->
             <!-- setting positions of all objects -->
             <a-box id="cube1" position='0 0.5 0' material='opacity: 1; side:double; color:red;' visible="true"></a-box>
             <a-sphere id="sphere1" position="0 0.5 0" material='opacity: 1'radius="0.5" color="#EF2D5E" visible="false"></a-sphere>
             <a-cylinder id="cylinder1" position="0 0.8 0" material='opacity:1;'radius="0.5" height="1.5" color="#FFC65D" visible="false"></a-cylinder>
             <a-box id="cube2" position='0 0.5 0' material='opacity: 0.5; color:red;' visible="false"></a-box>
             <a-sphere id="sphere2" position="0 0.5 0" material='opacity:0.5' radius="0.5" color="#EF2D5E" visible="false"></a-sphere>
             <a-cylinder id="cylinder2" position="0 0.8 0" material='opacity:0.5;'radius="0.5" height="1.5" color="#FFC65D" visible="false"></a-cylinder>
             <a-image id="spark" width="1" height="0.5" rotation="0 0 0" position="-1.5 0 0" src="#canvas"></a-image>
             <a-image id="cubed" width="1" height="3" rotation="0 0 0" position="2.5 0 0" src="#canvas1" visible="false"></a-image>
             <a-image id="sphered" width="1" height="3" rotation="0 0 0" position="2.5 0 0" src="#canvas2" visible="false"></a-image>
             <a-image id="cylindered" width="1" height="3" rotation="0 0 0" position="2.6 0 0" src="#canvas3" visible="false"></a-image>
             <a-image id="surfacearea" width="1" height="1" rotation="0 0 0" position="-1.5 -1.25 0" src="#canvas4" visible="false"></a-image>
             <a-image id="cubeans" width="2" height="1" rotation="0 0 0" position="0.25 -1.25 0" src="#canvas5" visible="false"></a-image>
             <a-image id="sphereans" width="2" height="1" rotation="0 0 0" position="0.25 -1.25 0" src="#canvas6" visible="false"></a-image>
             <a-image id="cylinderans" width="3" height="2" rotation="0 0 0" position="0.6 -1.2 0" src="#canvas7" visible="false"></a-image>
             <a-plane id="line1" position="0 1 0.5" depth="0.01" width="0.0" height="0.05" color="#00f936" rotation="-90 0 0"visible="false"></a-plane>
             <a-plane id="line2" position="0.25 0.5 0" depth="0.05" width="0.5" height="0.05" color="#00f936" rotation="-90 0 0"visible="false"></a-plane>
             <a-plane id="line3" position="0.25 0.8 0" depth="0.05" width="0.5" height="0.05" color="#00f936" rotation="-90 0 0"visible="false"></a-plane>
             <a-entity id="line4" line="start: 0.5 0.5 -5; end: 0.5 -3 -5; color: blue"></a-entity>
             <a-image id="a" width="0.5" height="0.25" rotation="0 0 0" position="-0.5 0 -5" src="#canvas8" visible="false"></a-image>
             <a-image id="r" width="0.5" height="0.25" rotation="0 0 0" position="-0.25 1.5 -3" src="#canvas9" visible="false"></a-image>
             <a-image id="h" width="0.5" height="0.25" rotation="0 0 0" position="0.85 -1.5 -5" src="#canvas10" visible="false"></a-image>
            
            
         </a-marker>
 
         <!-- use this <a-entity camera> to support multiple-markers, otherwise use <a-marker-camera> instead of </a-marker> -->
         <a-entity camera></a-entity>
         <a-cursor></a-cursor>
         </a-scene>
 </body>
 
```
- Run the code on localhost web browser with the following command
```
python3 -m http.server 8080(port number)
```
##  **Operation:**
- A hiro marker is used as the pattern
- Camera is pointed at hiro marker and shapes are formed on it. Surface area formula is displayed on clicking the solid, information about the solid along with examples is displayed on clicking the solid too. 
- On pressing the next arrow, the shape of the solid changes from cube to sphere to cylinder

## **Demo video link:**
https://www.dropbox.com/s/ni5kd24u9mmh2yw/volume.mkv?dl=0
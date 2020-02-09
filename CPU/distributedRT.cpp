// ======================================================================== //
// Copyright 2009-2020 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <embree3/rtcore.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <cstddef>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>


/*
 * A minimal tutorial.
 *
 * It demonstrates how to intersect a ray with a single triangle. It is
 * meant to get you started as quickly as possible, and does not output
 * an image.
 *
 * For more complex examples, see the other tutorials.
 *
 * Compile this file using
 *
 *   gcc -std=c99 \
 *       -I<PATH>/<TO>/<EMBREE>/include \
 *       -o minimal \
 *       minimal.c \
 *       -L<PATH>/<TO>/<EMBREE>/lib \
 *       -lembree3
 *
 * You should be able to compile this using a C or C++ compiler.
 */

/*
 * This is only required to make the tutorial compile even when
 * a custom namespace is set.
 */
#if defined(RTC_NAMESPACE_OPEN)
RTC_NAMESPACE_OPEN
#endif

glm::vec3 groundFaceColours[3] = {glm::vec3(0, 0, 0), glm::vec3(128, 128, 128), glm::vec3(64, 64, 64)};
glm::vec3 cubeFaceColours[12] = {glm::vec3(32, 0, 0), glm::vec3(0, 32, 0), glm::vec3(0, 0, 32),
                                 glm::vec3(64, 0, 0), glm::vec3(0, 64, 0), glm::vec3(0, 0, 64),
                                 glm::vec3(128, 0, 0), glm::vec3(0, 128, 0), glm::vec3(0, 0, 128),
                                 glm::vec3(255, 0, 0), glm::vec3(0, 255, 0), glm::vec3(0, 0, 255)};



/*
 * We will register this error handler with the device in initializeDevice(),
 * so that we are automatically informed on errors.
 * This is extremely helpful for finding bugs in your code, prevents you
 * from having to add explicit error checking to each Embree API call.
 */
void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
  printf("error %d: %s\n", error, str);
}


//error callback used by GLFW
void glfwErrorFunction(int error, const char* description) {
  fprintf(stderr, "Error: %s\n", description);
}

//key callback used to handle key press events
static void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, 1); //TODO: GLFW_TRUE HERE??
}


/*
 * Embree has a notion of devices, which are entities that can run
 * raytracing kernels.
 * We initialize our device here, and then register the error handler so that
 * we don't miss any errors.
 *
 * rtcNewDevice() takes a configuration string as an argument. See the API docs
 * for more information.
 *
 * Note that RTCDevice is reference-counted.
 */
RTCDevice initializeDevice()
{
  RTCDevice device = rtcNewDevice(NULL);

  if (!device)
    printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));

  rtcSetDeviceErrorFunction(device, errorFunction, NULL);
  return device;
}

/*
 * Create a scene, which is a collection of geometry objects. Scenes are
 * what the intersect / occluded functions work on. You can think of a
 * scene as an acceleration structure, e.g. a bounding-volume hierarchy.
 *
 * Scenes, like devices, are reference-counted.
 */
RTCScene initializeScene(RTCDevice device)
{
  RTCScene scene = rtcNewScene(device);

  /*
   * Create a triangle mesh geometry, and initialize a single triangle.
   * You can look up geometry types in the API documentation to
   * find out which type expects which buffers.
   *
   * We create buffers directly on the device, but you can also use
   * shared buffers. For shared buffers, special care must be taken
   * to ensure proper alignment and padding. This is described in
   * more detail in the API documentation.
   */
  RTCGeometry ground = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
  float* groundVertices = (float*) rtcSetNewGeometryBuffer(ground,
                                                     RTC_BUFFER_TYPE_VERTEX,
                                                     0,
                                                     RTC_FORMAT_FLOAT3,
                                                     3*sizeof(float),
                                                     4);

  unsigned* groundIndices = (unsigned*) rtcSetNewGeometryBuffer(ground,
                                                          RTC_BUFFER_TYPE_INDEX,
                                                          0,
                                                          RTC_FORMAT_UINT3,
                                                          3*sizeof(unsigned),
                                                          2);



  /* create face and vertex color arrays */
  //groundVertexColours = (glm::vec3*) alignedMalloc(4*sizeof(glm::vec3),16);

  // For buffer, need to be array of values  rather than vec3
  //Casting from vec3 more memory taken up, just write directly
  if (groundVertices && groundIndices)
  {
    groundVertices[0] = -10.f; groundVertices[1] = -2.f; groundVertices[2] = -10.f;
    groundVertices[3] = -10.f; groundVertices[4] = -2.f; groundVertices[5] = +10.f;
    groundVertices[6] = +10.f; groundVertices[7] = -2.f; groundVertices[8] = -10.f;
    groundVertices[9] = +10.f; groundVertices[10] = -2.f; groundVertices[11] = 10.f;

    groundIndices[0] = 0; groundIndices[1] = 1; groundIndices[2] = 2;
    groundIndices[3] = 1; groundIndices[4] = 3; groundIndices[5] = 2;

    //groundVertexColours[0] = glm::vec3(128,128,128);
    //groundVertexColours[1] = glm::vec3(128,128,128);
    //groundVertexColours[2] = glm::vec3(128,128,128);
    //groundVertexColours[3] = glm::vec3(128,128,128);

    //groundFaceColours[0] = glm::vec3(128,128,128);
    //groundFaceColours[1] = glm::vec3(128,128,128);
  }

  /*
   * You must commit geometry objects when you are done setting them up,
   * or you will not get any intersections.
   */
  rtcCommitGeometry(ground);

  /*
   * In rtcAttachGeometry(...), the scene takes ownership of the geom
   * by increasing its reference count. This means that we don't have
   * to hold on to the geom handle, and may release it. The geom object
   * will be released automatically when the scene is destroyed.
   *
   * rtcAttachGeometry() returns a geometry ID. We could use this to
   * identify intersected objects later on.
   */
  rtcAttachGeometry(scene, ground);
  rtcReleaseGeometry(ground);



  RTCGeometry cube = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
  float* cubeVertices = (float*) rtcSetNewGeometryBuffer(cube,
                                                     RTC_BUFFER_TYPE_VERTEX,
                                                     0,
                                                     RTC_FORMAT_FLOAT3,
                                                     3*sizeof(float),
                                                     8);

  unsigned* cubeIndices = (unsigned*) rtcSetNewGeometryBuffer(cube,
                                                          RTC_BUFFER_TYPE_INDEX,
                                                          0,
                                                          RTC_FORMAT_UINT3,
                                                          3*sizeof(unsigned),
                                                          12);

  // For buffer, need to be array of values  rather than vec3
  //Casting from vec3 more memory taken up, just write directly
  if (cubeVertices && cubeIndices)
  {
    cubeVertices[0] = -1; cubeVertices[1] = -1; cubeVertices[2] = -1;
    cubeVertices[3] = -1; cubeVertices[4] = -1; cubeVertices[5] = +1;
    cubeVertices[6] = -1; cubeVertices[7] = +1; cubeVertices[8] = -1;
    cubeVertices[9] = -1; cubeVertices[10] = +1; cubeVertices[11] = +1;
    cubeVertices[12] = +1; cubeVertices[13] = -1; cubeVertices[14] = -1;
    cubeVertices[15] = +1; cubeVertices[16] = -1; cubeVertices[17] = +1;
    cubeVertices[18] = +1; cubeVertices[19] = +1; cubeVertices[20] = -1;
    cubeVertices[21] = +1; cubeVertices[22] = +1; cubeVertices[23] = +1;

    cubeIndices[0] = 0; cubeIndices[1] = 1; cubeIndices[2] = 2;
    cubeIndices[3] = 1; cubeIndices[4] = 3; cubeIndices[5] = 2;
    cubeIndices[6] = 4; cubeIndices[7] = 6; cubeIndices[8] = 5;
    cubeIndices[9] = 5; cubeIndices[10] = 6; cubeIndices[11] = 7;
    cubeIndices[12] = 0; cubeIndices[13] = 4; cubeIndices[14] = 1;
    cubeIndices[15] = 1; cubeIndices[16] = 4; cubeIndices[17] = 5;
    cubeIndices[18] = 2; cubeIndices[19] = 3; cubeIndices[20] = 6;
    cubeIndices[21] = 3; cubeIndices[22] = 7; cubeIndices[23] = 6;
    cubeIndices[24] = 0; cubeIndices[25] = 2; cubeIndices[26] = 4;
    cubeIndices[27] = 2; cubeIndices[28] = 6; cubeIndices[29] = 4;
    cubeIndices[30] = 1; cubeIndices[31] = 5; cubeIndices[32] = 3;
    cubeIndices[33] = 3; cubeIndices[34] = 5; cubeIndices[35] = 7;
  }

  rtcCommitGeometry(cube);
  rtcAttachGeometry(scene, cube);
  rtcReleaseGeometry(cube);


  /*
   * Like geometry objects, scenes must be committed. This lets
   * Embree know that it may start building an acceleration structure.
   */
  rtcCommitScene(scene);

  return scene;
}

/*
 * Cast a single ray with origin (ox, oy, oz) and direction
 * (dx, dy, dz).
 */
glm::vec2 castRay(RTCScene scene,
             float ox, float oy, float oz,
             float dx, float dy, float dz)
{
  /*
   * The intersect context can be used to set intersection
   * filters or flags, and it also contains the instance ID stack
   * used in multi-level instancing.
   */
  struct RTCIntersectContext context;
  rtcInitIntersectContext(&context);

  /*
   * The ray hit structure holds both the ray and the hit.
   * The user must initialize it properly -- see API documentation
   * for rtcIntersect1() for details.
   */
  struct RTCRayHit rayhit;
  rayhit.ray.org_x = ox;
  rayhit.ray.org_y = oy;
  rayhit.ray.org_z = oz;
  rayhit.ray.dir_x = dx;
  rayhit.ray.dir_y = dy;
  rayhit.ray.dir_z = dz;
  rayhit.ray.tnear = 0;
  rayhit.ray.tfar = std::numeric_limits<float>::infinity();
  rayhit.ray.mask = 0;
  rayhit.ray.flags = 0;
  rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  /*
   * There are multiple variants of rtcIntersect. This one
   * intersects a single ray with the scene.
   */
  rtcIntersect1(scene, &context, &rayhit);

  //printf("%f, %f, %f: ", ox, oy, oz);
  if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
  {
    /* Note how geomID and primID identify the geometry we just hit.
     * We could use them here to interpolate geometry information,
     * compute shading, etc.
     * Since there is only a single triangle in this scene, we will
     * get geomID=0 / primID=0 for all hits.
     * There is also instID, used for instancing. See
     * the instancing tutorials for more information */
    //printf("Found intersection on geometry %d, primitive %d at tfar=%f\n",
    //       rayhit.hit.geomID,
    //       rayhit.hit.primID,
    //       rayhit.ray.tfar);
    return glm::vec2(rayhit.hit.geomID+1, rayhit.hit.primID+1);

  }
  else{
    //printf("Did not find any intersection.\n");

    return glm::vec2(0,0);
  }
}


/* -------------------------------------------------------------------------- */

int main()
{
  /* Initialization. All of this may fail, but we will be notified by
   * our errorFunction. */
  RTCDevice device = initializeDevice();
  RTCScene scene = initializeScene(device);

  int width  = 256;
  int height = 256;

  //specifies callback function to handle GLFW errors
  glfwSetErrorCallback(glfwErrorFunction);

  //initialises GLFW library
  glfwInit();

  //sets required OpenGL version
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,0);

  //returns handle to object which combines window and OpenGL context
  GLFWwindow* window = glfwCreateWindow(width,height,"distributedRT", NULL, NULL);

  //specifies call back function to handle GLFW key press events
  glfwSetKeyCallback(window, glfwKeyCallback);

  if (!window)
  {
    printf("Window or context creation failed.");
  }

  //sets current OpenGL context
  glfwMakeContextCurrent(window);

  //defines interval for vsync front and back buffer swap
  //default 0 leads to screen tearing
  glfwSwapInterval(1);

  //flag set to 1 on Alt-F4 or pressing close widget on title bar
  while (!glfwWindowShouldClose(window))
  {

    //frame buffer dimensions
    int fwidth;
    int fheight;
    glfwGetFramebufferSize(window, &fwidth, &fheight);


    glViewport(0, 0, fwidth, fheight);
    glClear(GL_COLOR_BUFFER_BIT);

    unsigned int data[fheight][fwidth][3];
    for (int i = 0; i < fheight; i++){
      for (int j = 0; j < fwidth; j++){
        int xdir = j - fwidth/2;
        int ydir = i - fheight/2;
        glm::vec2 hit = castRay(scene, 0, 0, -1.025, xdir, ydir, 1);
        glm::vec3 colour = glm::vec3(0, 0, 0);
        if (hit.x == 1) colour = groundFaceColours[(int)hit.y];
        if (hit.x == 2) colour = cubeFaceColours[(int)hit.y];
        data[i][j][0] = colour.x * 256 * 256 * 256;
        data[i][j][1] = colour.y * 256 * 256 * 256;
        data[i][j][2] = colour.z * 256 * 256 * 256;
      }
    }

    glDrawPixels(fwidth, fheight, GL_RGB, GL_UNSIGNED_INT, data);

    //swaps front and back buffer
    glfwSwapBuffers(window);
    // Continuously polls for events, rather than waiting for new input
    glfwPollEvents();
  }

  /* Though not strictly necessary in this example, you should
   * always make sure to release resources allocated through Embree. */
  rtcReleaseScene(scene);
  rtcReleaseDevice(device);

  //destroys window, handle becomes invalid
  glfwDestroyWindow(window);

  //releases resources allocated by GLFW
  glfwTerminate();

  return 0;
}

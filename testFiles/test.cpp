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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <limits>
#include <cstddef>
#include <GLFW/glfw3.h>


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

/* -------------------------------------------------------------------------- */

int main()
{
  /* Initialization. All of this may fail, but we will be notified by
   * our errorFunction. */
  int width  = 512;
  int height = 512;

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
        for(int k = 0; k < 3; k++){
          float randNum = rand()%255;
          data[i][j][k] = randNum * 256 * 256 * 256;
        }
      }
    }

    glDrawPixels(fwidth, fheight, GL_RGB, GL_UNSIGNED_INT, data);

    //swaps front and back buffer
    glfwSwapBuffers(window);
    // Continuously polls for events, rather than waiting for new input
    glfwPollEvents();
  }


  //destroys window, handle becomes invalid
  glfwDestroyWindow(window);

  //releases resources allocated by GLFW
  glfwTerminate();

  return 0;
}

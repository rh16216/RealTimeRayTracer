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
#include <iostream>
#include <fstream>

/* -------------------------------------------------------------------------- */

int main()
{

  int width  = 512;
  int height = 512;
  int max = 0;

  unsigned int data[height][width][3];
  for (int i = 0; i < height; i++){
    for (int j = 0; j < width; j++){
      for(int k = 0; k < 3; k++){
        float randNum = rand()%256;
        data[i][j][k] = randNum;
        if (max < randNum) max = randNum;
      }
    }
  }

 std::ofstream outFile;
 outFile.open ("output.ppm");
 outFile << "P3\n";
 outFile << width << " " << height << "\n";
 outFile << max << "\n";

 for (int i = 0; i < height; i++){
   for (int j = 0; j < width; j++){
     for(int k = 0; k < 3; k++){
       outFile << data[i][j][k] << " ";
     }
     outFile << " ";
   }
   outFile << "\n";
 }

 outFile.close();

  return 0;
}

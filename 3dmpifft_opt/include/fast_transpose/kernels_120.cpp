/*  This file is part of cuTranspose.

    cuTranspose is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    cuTranspose is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with cuTranspose.  If not, see <http://www.gnu.org/licenses/>.

    Copyright 2016 Ibai Gurrutxaga, Javier Muguerza, Jose L. Jodra.
*/

/********************************************
 * Includes                                 *
 ********************************************/
#include "fast_transpose.h"
#include "kernels_120.h"

/********************************************
 * Public functions                         *
 ********************************************/
__global__
void dev_transpose_120_ept1( data_t*       out,
                             const data_t* in,
                             int           np0,
                             int           np1,
                             int           np2 )
{

	__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];

	int x_in, y_in, z,
	    x_out, y_out,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x_in = lx + TILE_SIZE * bx;
	y_in = ly + TILE_SIZE * by;

	z = blockIdx.z;

	x_out = ly + TILE_SIZE * bx;
	y_out = lx + TILE_SIZE * by;

	ind_in = x_in + (y_in + z * np1) * np0;
	ind_out = y_out + (z + x_out * np2) * np1;

	if( x_in < np0 && y_in < np1 )
	{
		tile[lx][ly] = in[ind_in];
	}

	__syncthreads();

	if( y_out < np1 && x_out < np0 )
	{
		out[ind_out] = tile[ly][lx];
	}
}

__global__
void dev_transpose_120_ept2( data_t*       out,
                             const data_t* in,
                             int           np0,
                             int           np1,
                             int           np2 )
{

	__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];

	int x_in, y_in, z,
	    x_out, y_out,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x_in = lx + TILE_SIZE * bx;
	y_in = ly + TILE_SIZE * by;

	z = blockIdx.z;

	x_out = ly + TILE_SIZE * bx;
	y_out = lx + TILE_SIZE * by;

	ind_in = x_in + (y_in + z * np1) * np0;
	ind_out = y_out + (z + x_out * np2) * np1;

	if( x_in < np0 && y_in < np1 )
	{
		tile[lx][ly] = in[ind_in];
		if( y_in + 8 < np1 )
		{
			tile[lx][ly +  8] = in[ind_in +  8*np0];
		}
	}

	__syncthreads();

	if( y_out < np1 && x_out < np0 )
	{
		out[ind_out] = tile[ly][lx];
		if( x_out + 8 < np0 )
		{
			out[ind_out +  8*np1*np2] = tile[ly + 8][lx];
		}
	}

}

__global__
void dev_transpose_120_ept4( data_t*       out,
                             const data_t* in,
                             int           np0,
                             int           np1,
                             int           np2 )
{

	__shared__ data_t tile[TILE_SIZE][TILE_SIZE + 1];

	int x_in, y_in, z,
	    x_out, y_out,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x_in = lx + TILE_SIZE * bx;
	y_in = ly + TILE_SIZE * by;

	z = blockIdx.z;

	x_out = ly + TILE_SIZE * bx;
	y_out = lx + TILE_SIZE * by;

	ind_in = x_in + (y_in + z * np1) * np0;
	ind_out = y_out + (z + x_out * np2) * np1;

	if( x_in < np0 && y_in < np1 )
	{
		tile[lx][ly] = in[ind_in];
		if( y_in + 4 < np1 )
		{
			tile[lx][ly +  4] = in[ind_in +  4*np0];
			if( y_in + 8 < np1 )
			{
				tile[lx][ly +  8] = in[ind_in +  8*np0];
				if( y_in + 12 < np1 )
				{
					tile[lx][ly +  12] = in[ind_in +  12*np0];
				}
			}
		}
	}

	__syncthreads();

	if( y_out < np1 && x_out < np0 )
	{
		out[ind_out] = tile[ly][lx];
		if( x_out + 4 < np0 )
		{
			out[ind_out +  4*np1*np2] = tile[ly + 4][lx];
			if( x_out + 8 < np0 )
			{
				out[ind_out +  8*np1*np2] = tile[ly + 8][lx];
				if( x_out + 12 < np0 )
				{
					out[ind_out +  12*np1*np2] = tile[ly + 12][lx];
				}
			}
		}
	}
}

__global__
void dev_transpose_120_in_place( data_t* data,
                                 int     np0 )
{
	__shared__ data_t cube13[BRICK_SIZE][BRICK_SIZE][BRICK_SIZE + 1];
	__shared__ data_t cube2[BRICK_SIZE][BRICK_SIZE][BRICK_SIZE + 1];

	int x1, y1, z1,
	    x2, y2, z2,
	    x3, y3, z3,
	    ind1, ind2, ind3;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    lz = threadIdx.z,
	    bx = blockIdx.x,
	    by = blockIdx.y,
	    bz = blockIdx.z;

	int diagonal = (bx == by && by == bz);

	if( bx > by || bx > bz ||
	    ((bx == by || bx == bz) && by > bz) )
		return;

	x1 = lx + BRICK_SIZE * bx;
	y1 = ly + BRICK_SIZE * by;
	z1 = lz + BRICK_SIZE * bz;

	x2 = ly + BRICK_SIZE * bx;
	y2 = lx + BRICK_SIZE * by;
	z2 = lz + BRICK_SIZE * bz;

	x3 = lz + BRICK_SIZE * bx;
	y3 = ly + BRICK_SIZE * by;
	z3 = lx + BRICK_SIZE * bz;

	ind1 = x1 + (y1 + z1 * np0) * np0;
	ind2 = y2 + (z2 + x2 * np0) * np0;
	ind3 = z3 + (x3 + y3 * np0) * np0;

	// Swap lx and ly to avoid the synchronization commented below.
	if( x1 < np0 && y1 < np0 && z1 < np0 )
		cube13[ly][lx][lz] = data[ind1];
	if( x2 < np0 && y2 < np0 && z2 < np0 )
		if( ! diagonal )
			cube2[lx][ly][lz] = data[ind2];

	__syncthreads();

	if( diagonal )
	{
		if( x1 < np0 && y1 < np0 && z1 < np0 )
			data[ind1] = cube13[lx][lz][ly];
	}
	else
	{
		if( x2 < np0 && y2 < np0 && z2 < np0 )
			data[ind2] = cube13[lx][ly][lz];

		//__syncthreads();
		if( x3 < np0 && y3 < np0 && z3 < np0 )
		{
			cube13[lx][ly][lz] = data[ind3];
			data[ind3] = cube2[ly][lz][lx];
		}
		__syncthreads();

		if( x1 < np0 && y1 < np0 && z1 < np0 )
			data[ind1] = cube13[lz][ly][lx];
	}
}

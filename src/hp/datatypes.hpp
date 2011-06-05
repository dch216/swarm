/*************************************************************************
 * Copyright (C) 2011 by Saleh Dindar and the Swarm-NG Development Team  *
 *                                                                       *
 * This program is free software; you can redistribute it and/or modify  *
 * it under the terms of the GNU General Public License as published by  *
 * the Free Software Foundation; either version 3 of the License.        *
 *                                                                       *
 * This program is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 * GNU General Public License for more details.                          *
 *                                                                       *
 * You should have received a copy of the GNU General Public License     *
 * along with this program; if not, write to the                         *
 * Free Software Foundation, Inc.,                                       *
 * 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ************************************************************************/

#pragma once

#define GPUAPI inline __device__ __host__


namespace swarm {
namespace hp {

/**
 * Array of structures with coalecsed access.
 * The Item should provide WARPSIZE and scalar_t
 * for offset calculation
 */
template<class Item, typename _Scalar = typename Item::scalar_t, int _WARPSIZE = Item::WARPSIZE >
struct CoalescedStructArray {
	Item * _array;
	size_t _block_count;
	static const int WARPSIZE = _WARPSIZE;
	typedef _Scalar scalar_t;

	GPUAPI CoalescedStructArray () {};
	GPUAPI CoalescedStructArray(Item * array, size_t block_count)
		:_array(array),_block_count(block_count){}
	GPUAPI Item& operator[] ( const int & i ) {
		size_t block_idx = i / WARPSIZE;
		size_t idx = i % WARPSIZE;
		scalar_t * blockaddr = (scalar_t*) (_array + block_idx);
		return * (Item *) ( blockaddr + idx );
	}
	GPUAPI int block_count()const{
		return _block_count ;
	}
	GPUAPI int size()const{
		return _block_count * WARPSIZE;
	}
	GPUAPI Item * get() {
		return _array;
	}

	GPUAPI Item * begin() {
		return _array;
	}

	GPUAPI Item * end() {
		return _array + _block_count;
	}
};


}
}

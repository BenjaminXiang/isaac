/* Copyright 2015-2017 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef ISAAC_DRIVER_STREAM_H
#define ISAAC_DRIVER_STREAM_H

#include <map>
#include "isaac/driver/context.h"
#include "isaac/driver/device.h"
#include "isaac/driver/handle.h"
#include "isaac/driver/buffer.h"

namespace isaac
{

namespace driver
{

class Kernel;
class Event;
class Range;
class Buffer;

// Command Queue
class Stream: public HandleInterface<Stream, CUstream>
{
public:
  //Constructors
  Stream(CUstream stream, bool take_ownership);
  Stream(Context const & context);

  //Accessors
  Handle<CUstream> const & cu() const;
  Context const & context() const;

  //Synchronize
  void synchronize();

  //Enqueue
  void enqueue(Kernel const & kernel, std::array<size_t, 3> grid, std::array<size_t, 3> block, std::vector<Event> const * = NULL, Event *event = NULL);

  // Write
  void write(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr);

  template<class T> void write(Buffer const & buffer, bool blocking, std::size_t offset, std::vector<T> const & x)
  { write(buffer, blocking, offset, x.size()*sizeof(T), x.data()); }

  // Read
  void read(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr);

  template<class T> void read(Buffer const & buffer, bool blocking, std::size_t offset, std::vector<T>& x)
  { read(buffer, blocking, offset, x.size()*sizeof(T), x.data()); }
private:
  Context context_;
  Handle<CUstream> cu_;
};


}

}

#endif
